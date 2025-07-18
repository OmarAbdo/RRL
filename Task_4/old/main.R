library(keras)
library(quantmod)
library(tensorflow)
if (!requireNamespace("progress", quietly = TRUE)) install.packages("progress")
library(progress)

# Dynamic working directory resolution
if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
  this_file <- rstudioapi::getActiveDocumentContext()$path
} else {
  args <- commandArgs(trailingOnly = FALSE)
  this_file <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
}
setwd(dirname(this_file))

## HYPER-PARAMETERS  -----------------------------------------------------
# --- Battery Parameters (from PDF) ---
battery_capacity_mwh <- 1.0 # Usable energy: 1 MWh
power_limit_mw <- 0.5 # Power limit: 0.5 MW
efficiency <- 0.9 # Round-trip efficiency: 90%
trade_cost <- 20 # Degradation cost: 20 EUR/MWh throughput
interval_hours <- 0.25 # Time interval: 15 minutes (0.25 hours)
energy_per_step_mwh <- power_limit_mw * interval_hours # Max energy per step

# --- RL Agent & Training Parameters (not specified in PDF) ---
window_size <- 10 # Look-back window for state representation
state_dim <- window_size + 2 # State vector: price window + current SoC + current price
hidden_units <- 32 # Network size (reduced for quick training)
episodes <- 10 # Training episodes (reduced for quick training)
replay_capacity <- 2000 # Max size of replay buffer (reduced for quick training)
warmup_mem <- 200 # Steps before training starts (reduced for quick training)
learning_rate <- 1e-3 # Adam optimizer learning rate
gamma <- 0.99 # Discount factor for future rewards
batch_size <- 32 # Replay buffer batch size
target_sync_freq <- 10 # Episodes before target network sync
eps_start <- 1.0 # Epsilon-greedy starting value
eps_final <- 0.05 # Epsilon-greedy final value
per_alpha <- 0.6 # PER hyperparameter: controls how much prioritization is used
per_beta <- 0.4 # PER hyperparameter: importance-sampling, from 0.4 to 1.0
per_epsilon <- 1e-6 # PER hyperparameter: small value to ensure all transitions have a non-zero priority


## LOAD DATA  ------------------------------------------------------------
# Load the data from the CSV file
data <- read.csv("data_task4.csv")

# Parse the DateTime column
data$DateTime.UTC. <- as.POSIXct(data$DateTime.UTC., format = "%Y-%m-%dT%H:%M:%SZ")

# Reverse the data to be in chronological order
data <- data[rev(1:nrow(data)), ]

# Extract prices
price <- data$Price.Currency.MWh.

# Calculate the price changes (returns) as a percentage for state representation
rets <- na.omit(diff(price) / price[-length(price)]) * 100 # % price changes

# Normalize prices for the state representation to stabilize learning
price_norm <- (price - mean(price)) / sd(price)

# training and test split
train_size <- floor(0.8 * length(rets))
rets_train <- rets[1:train_size]
rets_test <- rets[(train_size + 1):length(rets)]

rets <- rets_train # Use training returns for state in training loop
prices_for_env <- price[1:length(rets_train)] # Use training prices for environment in training loop
price_norm_for_env <- price_norm[1:length(rets_train)] # Use normalized prices for state in training loop
n_obs <- length(rets)
max_steps_ep <- n_obs - window_size - 1 # guard against infinite loops

## STATE: # t is  (window_size : n_obs-1) and current battery state of charge (0-1)
make_state <- function(t, soc) {
  # Use normalized price in state for better learning
  current_price_norm <- price_norm_for_env[t]
  c(as.numeric(rets[(t - window_size + 1):t]), soc, current_price_norm)
}


## ENVIRONMENT HELPERS  --------------------------------------------------
env_reset <- function() {
  list(
    t = window_size, soc = 0.5, # Initial SoC at 50%
    equity = 1.0, done = FALSE
  )
}

env_step <- function(env, action) {
  t2 <- env$t + 1
  soc <- env$soc
  equity <- env$equity

  # Get current price (raw price for reward calculation)
  current_price <- prices_for_env[t2]

  # Initialize reward
  reward <- 0

  # Correct efficiency application (sqrt applied at each stage for round-trip)
  sqrt_efficiency <- sqrt(efficiency)

  # Action: 0 = Hold, 1 = Charge, 2 = Discharge
  if (action == 1) { # Charge
    # How much can we charge?
    charge_amount <- min(energy_per_step_mwh, battery_capacity_mwh - soc * battery_capacity_mwh)
    # Update SoC considering charging efficiency
    soc_next <- soc + (charge_amount * sqrt_efficiency) / battery_capacity_mwh
    # BUG FIX: Cost calculation should not be divided by 1000. Prices and energy are both in MWh.
    cost <- charge_amount * current_price
    degradation_cost <- charge_amount * trade_cost
    reward <- -cost - degradation_cost
  } else if (action == 2) { # Discharge
    # How much can we discharge?
    discharge_amount <- min(energy_per_step_mwh, soc * battery_capacity_mwh)
    # Energy that actually gets sold after discharge efficiency
    energy_sold <- discharge_amount * sqrt_efficiency
    # Update SoC
    soc_next <- soc - discharge_amount / battery_capacity_mwh
    # BUG FIX: Revenue calculation should not be divided by 1000.
    revenue <- energy_sold * current_price
    degradation_cost <- discharge_amount * trade_cost
    reward <- revenue - degradation_cost
  } else { # Hold
    soc_next <- soc
    reward <- 0
  }

  # Ensure SoC is within bounds [0, 1]
  soc_next <- max(0, min(1, soc_next))

  # Update equity based on reward
  equity_next <- equity + reward

  done <- (t2 >= n_obs - 1)

  list(
    next_env = list(
      t = t2, soc = soc_next,
      equity = equity_next, done = done
    ),
    reward = reward,
    obs = make_state(t2, soc_next)
  )
}


## BUILD Q-NETWORK + TARGET NETWORK  -------------------------------------
build_qnet <- function() {
  input <- layer_input(shape = state_dim)
  hidden <- input %>% layer_dense(hidden_units, activation = "relu")
  output <- hidden %>% layer_dense(3, activation = "linear") # 3 outputs for Hold, Charge, Discharge
  keras_model(input, output) %>%
    compile(optimizer = tf$keras$optimizers$legacy$Adam(learning_rate = learning_rate), loss = "mse")
}

## REPLAY BUFFER  --------------------------------------------------------
replay <- new.env(parent = emptyenv())
replay$S <- array(0, dim = c(replay_capacity, state_dim))
replay$A <- integer(replay_capacity)
replay$R <- numeric(replay_capacity)
replay$S2 <- array(0, dim = c(replay_capacity, state_dim))
replay$D <- integer(replay_capacity)
replay$idx <- 1L
replay$full <- FALSE

store_transition <- function(s, a, r, s2, done) {
  i <- replay$idx
  replay$S[i, ] <- s
  replay$A[i] <- a
  replay$R[i] <- r
  replay$S2[i, ] <- s2
  replay$D[i] <- done
  replay$idx <- i %% replay_capacity + 1L
  if (i == replay_capacity) replay$full <- TRUE
}

sample_batch <- function(size) {
  max_i <- if (replay$full) replay_capacity else (replay$idx - 1L)
  idx <- sample.int(max_i, size)
  list(
    S = replay$S[idx, , drop = FALSE],
    A = replay$A[idx],
    R = replay$R[idx],
    S2 = replay$S2[idx, , drop = FALSE],
    D = replay$D[idx]
  )
}

## PER BUFFER -----------------------------------------------------------
per_replay <- new.env(parent = emptyenv())
per_replay$S <- array(0, dim = c(replay_capacity, state_dim))
per_replay$A <- integer(replay_capacity)
per_replay$R <- numeric(replay_capacity)
per_replay$S2 <- array(0, dim = c(replay_capacity, state_dim))
per_replay$D <- integer(replay_capacity)
per_replay$priorities <- numeric(replay_capacity)
per_replay$idx <- 1L
per_replay$full <- FALSE

store_transition_per <- function(s, a, r, s2, done) {
  i <- per_replay$idx
  per_replay$S[i, ] <- s
  per_replay$A[i] <- a
  per_replay$R[i] <- r
  per_replay$S2[i, ] <- s2
  per_replay$D[i] <- done

  # Set max priority for new experience, handle case where buffer is empty
  max_priority <- if (per_replay$full || per_replay$idx > 1) max(per_replay$priorities[1:max(1, per_replay$idx - 1)]) else 1.0
  if (is.infinite(max_priority)) max_priority <- 1.0
  per_replay$priorities[i] <- max_priority

  per_replay$idx <- i %% replay_capacity + 1L
  if (i == replay_capacity) per_replay$full <- TRUE
}

sample_batch_per <- function(size, alpha, beta) {
  max_i <- if (per_replay$full) replay_capacity else (per_replay$idx - 1L)
  priorities <- per_replay$priorities[1:max_i]

  probs <- priorities^alpha / sum(priorities^alpha)

  idx <- sample.int(max_i, size, prob = probs, replace = TRUE)

  weights <- (max_i * probs[idx])^-beta
  weights <- weights / max(weights, na.rm = TRUE) # Normalize for stability

  list(
    S = per_replay$S[idx, , drop = FALSE],
    A = per_replay$A[idx],
    R = per_replay$R[idx],
    S2 = per_replay$S2[idx, , drop = FALSE],
    D = per_replay$D[idx],
    indices = idx,
    weights = weights
  )
}

update_priorities <- function(indices, td_errors, epsilon) {
  per_replay$priorities[indices] <- abs(td_errors) + epsilon
}

eps_decay_rate <- log(eps_start / eps_final) / (episodes - 1)

epsilon <- function(ep) {
  eps_start * exp(-eps_decay_rate * (ep - 1))
}

## T1: Baseline Agent Training and Saving --------------------------------------------------------
# Re-initialize qnet and target_net for baseline training
qnet_baseline <- build_qnet()
target_net_baseline <- build_qnet()
target_net_baseline$set_weights(qnet_baseline$get_weights())

cat("\nStarting Baseline Agent Training...\n")
pb_baseline <- progress_bar$new(
  format = "  Episode :ep/:total [:bar] :percent in :elapsed | ETA: :eta",
  total = episodes, clear = FALSE, width = 60
)
for (ep in 1:episodes) {
  env <- env_reset()
  state <- make_state(env$t, env$soc) # Changed env$pos to env$soc
  eps <- epsilon(ep)
  ep_reward <- 0

  pb_steps_baseline <- progress_bar$new(
    format = "    Step :step/:total [:bar] :percent",
    total = max_steps_ep, clear = FALSE, width = 50
  )

  for (step in 1:max_steps_ep) {
    pb_steps_baseline$tick()
    # choose action
    if (runif(1) < eps) {
      action <- sample(0:2, 1)
    } # explore (0: Hold, 1: Charge, 2: Discharge)
    else {
      action <- which.max(predict(qnet_baseline, matrix(state, nrow = 1), verbose = 0)) - 1
    } # exploit

    # interact
    res <- env_step(env, action)
    store_transition(state, action, res$reward, res$obs, res$next_env$done)

    state <- res$obs
    env <- res$next_env
    ep_reward <- ep_reward + res$reward
    if (env$done) break

    # learn
    if ((replay$full || replay$idx > warmup_mem)) {
      batch <- sample_batch(batch_size)
      q_next <- predict(target_net_baseline, batch$S2, verbose = 0)
      q_max <- apply(q_next, 1, max)
      y <- batch$R + (1 - batch$D) * gamma * q_max
      y_keras <- predict(qnet_baseline, batch$S, verbose = 0)
      y_keras[cbind(seq_len(batch_size), batch$A + 1)] <- y
      qnet_baseline %>% train_on_batch(batch$S, y_keras)
    }
  }

  # target network sync
  if (ep %% target_sync_freq == 0) {
    target_net_baseline$set_weights(qnet_baseline$get_weights())
  }

  pb_baseline$tick()
  cat(sprintf(
    " | Episode Reward: %.4f | Equity: %.3f\n",
    ep_reward, env$equity
  ))
}

# Save the trained baseline network weights (T5)
save_model_weights_hdf5(qnet_baseline, "model_weights_baseline.h5")
cat("Baseline agent model weights saved to model_weights_baseline.h5\n")


## T4: Mandatory Enhancement (DDQN) Training and Saving --------------------------------------------------------
# Re-initialize qnet and target_net for DDQN training
qnet_ddqn <- build_qnet()
target_net_ddqn <- build_qnet()
target_net_ddqn$set_weights(qnet_ddqn$get_weights())

cat("\nStarting DDQN Agent Training...\n")
pb_ddqn <- progress_bar$new(
  format = "  Episode :ep/:total [:bar] :percent in :elapsed | ETA: :eta",
  total = episodes, clear = FALSE, width = 60
)
for (ep in 1:episodes) {
  env <- env_reset()
  state <- make_state(env$t, env$soc)
  eps <- epsilon(ep)
  ep_reward <- 0

  pb_steps_ddqn <- progress_bar$new(
    format = "    Step :step/:total [:bar] :percent",
    total = max_steps_ep, clear = FALSE, width = 50
  )

  for (step in 1:max_steps_ep) {
    pb_steps_ddqn$tick()
    # choose action
    if (runif(1) < eps) {
      action <- sample(0:2, 1)
    } # explore (0: Hold, 1: Charge, 2: Discharge)
    else {
      action <- which.max(predict(qnet_ddqn, matrix(state, nrow = 1), verbose = 0)) - 1
    } # exploit

    # interact
    res <- env_step(env, action)
    store_transition(state, action, res$reward, res$obs, res$next_env$done)

    state <- res$obs
    env <- res$next_env
    ep_reward <- ep_reward + res$reward
    if (env$done) break

    # learn
    if ((replay$full || replay$idx > warmup_mem)) {
      batch <- sample_batch(batch_size)
      # Double DQN: Use online_net to select action, target_net to evaluate Q-value
      q_online_next <- predict(qnet_ddqn, batch$S2, verbose = 0)
      best_actions <- apply(q_online_next, 1, which.max) - 1 # Get the action indices (0, 1, or 2)

      q_target_next <- predict(target_net_ddqn, batch$S2, verbose = 0)
      q_max <- q_target_next[cbind(seq_len(batch_size), best_actions + 1)]

      y <- batch$R + (1 - batch$D) * gamma * q_max
      y_keras <- predict(qnet_ddqn, batch$S, verbose = 0) # this is only needed to generate q values for actions not taken
      y_keras[cbind(seq_len(batch_size), batch$A + 1)] <- y # crucial mostly for keras, we set the y value only for taken actions to get their loss (y-q)^2 and leave the rest to the original q-values, essentially setting their loss to be 0 (q-q)^2
      qnet_ddqn %>% train_on_batch(batch$S, y_keras) # y_keras is the artificial "true" target, so we train the qnet to predict these values
    }
  }

  # target network sync
  if (ep %% target_sync_freq == 0) {
    target_net_ddqn$set_weights(qnet_ddqn$get_weights())
  }

  pb_ddqn$tick()
  cat(sprintf(
    " | Episode Reward: %.4f | Equity: %.3f\n",
    ep_reward, env$equity
  ))
}

# Save the trained DDQN model weights (T5)
save_model_weights_hdf5(qnet_ddqn, "model_weights_ddqn.h5")
cat("Trained DDQN model weights saved to model_weights_ddqn.h5\n")

## T4: Mandatory Enhancement (PER) Training and Saving --------------------------------------------------------
qnet_per <- build_qnet()
target_net_per <- build_qnet()
target_net_per$set_weights(qnet_per$get_weights())

cat("\nStarting PER Agent Training...\n")
pb_per <- progress_bar$new(
  format = "  Episode :ep/:total [:bar] :percent in :elapsed | ETA: :eta",
  total = episodes, clear = FALSE, width = 60
)
for (ep in 1:episodes) {
  env <- env_reset()
  state <- make_state(env$t, env$soc)
  eps <- epsilon(ep)
  ep_reward <- 0

  for (step in 1:max_steps_ep) {
    # Action selection
    if (runif(1) < eps) {
      action <- sample(0:2, 1)
    } else {
      action <- which.max(predict(qnet_per, matrix(state, nrow = 1), verbose = 0)) - 1
    }

    # Interact with environment
    res <- env_step(env, action)
    store_transition_per(state, action, res$reward, res$obs, res$next_env$done)

    state <- res$obs
    env <- res$next_env
    ep_reward <- ep_reward + res$reward
    if (env$done) break

    # Learn
    if (per_replay$full || per_replay$idx > warmup_mem) {
      batch <- sample_batch_per(batch_size, per_alpha, per_beta)

      q_next <- predict(target_net_per, batch$S2, verbose = 0)
      q_max <- apply(q_next, 1, max)

      y <- batch$R + (1 - batch$D) * gamma * q_max

      y_keras <- predict(qnet_per, batch$S, verbose = 0)

      td_errors <- y - y_keras[cbind(seq_len(batch_size), batch$A + 1)]

      y_keras[cbind(seq_len(batch_size), batch$A + 1)] <- y

      qnet_per %>% train_on_batch(batch$S, y_keras, sample_weight = batch$weights)

      update_priorities(batch$indices, td_errors, per_epsilon)
    }
  }

  if (ep %% target_sync_freq == 0) {
    target_net_per$set_weights(qnet_per$get_weights())
  }

  pb_per$tick()
  cat(sprintf(" | Episode Reward: %.4f | Equity: %.3f\n", ep_reward, env$equity))
}

save_model_weights_hdf5(qnet_per, "model_weights_per.h5")
cat("Trained PER model weights saved to model_weights_per.h5\n")


## T2: Benchmarks & T3: Evaluation --------------------------------------------------------
# Load the trained baseline model for evaluation
qnet_baseline_eval <- build_qnet()
load_model_weights_hdf5(qnet_baseline_eval, "model_weights_baseline.h5")

# Load the trained DDQN model for evaluation
qnet_ddqn_eval <- build_qnet()
load_model_weights_hdf5(qnet_ddqn_eval, "model_weights_ddqn.h5")

# Load the trained PER model for evaluation
qnet_per_eval <- build_qnet()
load_model_weights_hdf5(qnet_per_eval, "model_weights_per.h5")

# Re-run the trained DDQN agent to get its equity curve
env_ddqn_eval <- env_reset()
state_ddqn_eval <- make_state(env_ddqn_eval$t, env_ddqn_eval$soc)
ddqn_equity_curve <- c(env_ddqn_eval$equity)

for (step in 1:max_steps_ep) {
  action <- which.max(predict(qnet_ddqn_eval, matrix(state_ddqn_eval, nrow = 1), verbose = 0)) - 1L
  res <- env_step(env_ddqn_eval, action)
  state_ddqn_eval <- res$obs
  env_ddqn_eval <- res$next_env
  ddqn_equity_curve <- c(ddqn_equity_curve, env_ddqn_eval$equity)
  if (env_ddqn_eval$done) break
}

# Re-run the trained PER agent to get its equity curve
env_agent <- env_reset()
state_agent <- make_state(env_agent$t, env_agent$soc)
agent_equity_curve <- c(env_agent$equity)

for (step in 1:max_steps_ep) {
  action <- which.max(predict(qnet_per_eval, matrix(state_agent, nrow = 1), verbose = 0)) - 1L
  res <- env_step(env_agent, action)
  state_agent <- res$obs
  env_agent <- res$next_env
  agent_equity_curve <- c(agent_equity_curve, env_agent$equity)
  if (env_agent$done) break
}

# Re-run the trained Baseline agent to get its equity curve
env_baseline_eval <- env_reset()
state_baseline_eval <- make_state(env_baseline_eval$t, env_baseline_eval$soc)
baseline_equity_curve <- c(env_baseline_eval$equity)

for (step in 1:max_steps_ep) {
  action <- which.max(predict(qnet_baseline_eval, matrix(state_baseline_eval, nrow = 1), verbose = 0)) - 1L
  res <- env_step(env_baseline_eval, action)
  state_baseline_eval <- res$obs
  env_baseline_eval <- res$next_env
  baseline_equity_curve <- c(baseline_equity_curve, env_baseline_eval$equity)
  if (env_baseline_eval$done) break
}

# (a) Random Policy
env_random <- env_reset()
state_random <- make_state(env_random$t, env_random$soc)
random_equity_curve <- c(env_random$equity)

for (step in 1:max_steps_ep) {
  action <- sample(0:2, 1) # Random action: 0=Hold, 1=Charge, 2=Discharge
  res <- env_step(env_random, action)
  state_random <- res$obs
  env_random <- res$next_env
  random_equity_curve <- c(random_equity_curve, env_random$equity)
  if (env_random$done) break
}

# (b) Heuristic Policy: Charge when price is negative, Discharge when price is positive
env_heuristic <- env_reset()
state_heuristic <- make_state(env_heuristic$t, env_heuristic$soc)
heuristic_equity_curve <- c(env_heuristic$equity)

for (step in 1:max_steps_ep) {
  current_price_for_heuristic <- prices_for_env[env_heuristic$t + 1]
  action <- 0 # Default to Hold
  if (current_price_for_heuristic < 0) {
    action <- 1 # Charge
  } else if (current_price_for_heuristic > 0) {
    action <- 2 # Discharge
  }
  res <- env_step(env_heuristic, action)
  state_heuristic <- res$obs
  env_heuristic <- res$next_env
  heuristic_equity_curve <- c(heuristic_equity_curve, env_heuristic$equity)
  if (env_heuristic$done) break
}

# Plotting
max_len <- max(length(agent_equity_curve), length(random_equity_curve), length(heuristic_equity_curve), length(baseline_equity_curve), length(ddqn_equity_curve))

plot_df <- data.frame(
  Step = 1:max_len,
  PER_Agent = c(agent_equity_curve, rep(NA, max_len - length(agent_equity_curve))),
  DDQN_Agent = c(ddqn_equity_curve, rep(NA, max_len - length(ddqn_equity_curve))),
  Baseline_Agent = c(baseline_equity_curve, rep(NA, max_len - length(baseline_equity_curve))),
  Random = c(random_equity_curve, rep(NA, max_len - length(random_equity_curve))),
  Heuristic = c(heuristic_equity_curve, rep(NA, max_len - length(heuristic_equity_curve)))
)

library(ggplot2)

ggplot(plot_df, aes(x = Step)) +
  geom_line(aes(y = PER_Agent, colour = "PER Agent")) +
  geom_line(aes(y = DDQN_Agent, colour = "DDQN Agent")) +
  geom_line(aes(y = Baseline_Agent, colour = "Baseline Agent")) +
  geom_line(aes(y = Random, colour = "Random Policy")) +
  geom_line(aes(y = Heuristic, colour = "Heuristic Policy")) +
  labs(title = "Battery Agent vs. Benchmarks", y = "Equity", x = "Time Steps") +
  scale_colour_manual("", values = c("PER Agent" = "blue", "DDQN Agent" = "orange", "Baseline Agent" = "purple", "Random Policy" = "red", "Heuristic Policy" = "green")) +
  theme_minimal()
