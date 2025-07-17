library(keras)
library(quantmod)
library(tensorflow)
if (!requireNamespace("progress", quietly = TRUE)) install.packages("progress")
library(progress)

## HYPER-PARAMETERS  -----------------------------------------------------
# For quick training on a Core i7 CPU (minutes instead of hours), consider these alternative configurations:
# episodes          <- 10 # Fewer episodes
# replay_capacity   <- 1000 # Smaller replay buffer
# warmup_mem        <- 100 # Faster warmup
# hidden_units      <- 16 # Smaller network

window_size <- 10 # look-back bars → state vector length
state_dim <- window_size + 1 # returns window  +  current position
hidden_units <- 16 #  32                 ## lower for quick training
learning_rate <- 1e-3
gamma <- 0.99 # discount
episodes <- 5 # 100       # +/- depending on patience ## lower for quick training

batch_size <- 32
replay_capacity <- 1000 # 5000   ## lower for quick training
warmup_mem <- 50 # 200    ## no training until buffer this large ## lower for quick training
target_sync_freq <- 10 # episodes

eps_start <- 1.0 # ε-greedy schedule
eps_final <- 0.05

trade_cost <- 0.02

# Battery parameters
battery_capacity_mwh <- 1.0
power_limit_mw <- 0.5
efficiency <- 0.9
interval_hours <- 0.25 # 15 minutes
energy_per_step_mwh <- power_limit_mw * interval_hours


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

# training and test split
train_size <- floor(0.8 * length(rets))
rets_train <- rets[1:train_size]
rets_test <- rets[(train_size + 1):length(rets)]

rets <- rets_train # Use training returns for state in training loop
prices_for_env <- price[1:length(rets_train)] # Use training prices for environment in training loop
n_obs <- length(rets)
max_steps_ep <- n_obs - window_size - 1 # guard against infinite loops

## STATE: # t is  (window_size : n_obs-1) and current battery state of charge (0-1)
make_state <- function(t, soc) {
  c(as.numeric(rets[(t - window_size + 1):t]), soc)
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

  # Get current price
  current_price <- prices_for_env[t2]

  # Initialize reward and energy moved
  reward <- 0
  energy_moved <- 0

  # Action: 0 = Hold, 1 = Charge, 2 = Discharge
  if (action == 1) { # Charge
    charge_amount <- min(energy_per_step_mwh, battery_capacity_mwh - soc * battery_capacity_mwh)
    charge_amount_eff <- charge_amount * efficiency
    soc_next <- soc + charge_amount_eff / battery_capacity_mwh
    cost <- charge_amount * current_price / 1000 # Convert MWh to kWh for price
    degradation_cost <- charge_amount * trade_cost
    reward <- -cost - degradation_cost
    energy_moved <- charge_amount
  } else if (action == 2) { # Discharge
    discharge_amount <- min(energy_per_step_mwh, soc * battery_capacity_mwh)
    discharge_amount_eff <- discharge_amount * efficiency
    soc_next <- soc - discharge_amount / battery_capacity_mwh
    revenue <- discharge_amount_eff * current_price / 1000 # Convert MWh to kWh for price
    degradation_cost <- discharge_amount * trade_cost
    reward <- revenue - degradation_cost
    energy_moved <- discharge_amount
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
  max_steps_ep <- 1000 # Reduced for quick training
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
qnet <- build_qnet()
target_net <- build_qnet()
target_net$set_weights(qnet$get_weights())

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
      action <- which.max(predict(qnet, matrix(state, nrow = 1), verbose = 0)) - 1
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
      q_online_next <- predict(qnet, batch$S2, verbose = 0)
      best_actions <- apply(q_online_next, 1, which.max) - 1 # Get the action indices (0, 1, or 2)

      q_target_next <- predict(target_net, batch$S2, verbose = 0)
      q_max <- q_target_next[cbind(seq_len(batch_size), best_actions + 1)]

      y <- batch$R + (1 - batch$D) * gamma * q_max
      y_keras <- predict(qnet, batch$S, verbose = 0) # this is only needed to generate q values for actions not taken
      y_keras[cbind(seq_len(batch_size), batch$A + 1)] <- y # crucial mostly for keras, we set the y value only for taken actions to get their loss (y-q)^2 and leave the rest to the original q-values, essentially setting their loss to be 0 (q-q)^2
      qnet %>% train_on_batch(batch$S, y_keras) # y_keras is the artificial "true" target, so we train the qnet to predict these values
    }
  }

  # target network sync
  if (ep %% target_sync_freq == 0) {
    target_net$set_weights(qnet$get_weights())
  }

  pb_ddqn$tick()
  cat(sprintf(
    " | Episode Reward: %.4f | Equity: %.3f\n",
    ep_reward, env$equity
  ))
}

# Save the trained DDQN model weights (T5)
save_model_weights_hdf5(qnet, "model_weights_ddqn.h5")
cat("Trained DDQN model weights saved to model_weights_ddqn.h5
")


## T2: Benchmarks & T3: Evaluation --------------------------------------------------------
# Load the trained baseline model for evaluation
qnet_baseline_eval <- build_qnet()
load_model_weights_hdf5(qnet_baseline_eval, "model_weights_baseline.h5")

# Load the trained DDQN model for evaluation
qnet_ddqn_eval <- build_qnet()
load_model_weights_hdf5(qnet_ddqn_eval, "model_weights_ddqn.h5")

# Re-run the trained DDQN agent to get its equity curve
env_agent <- env_reset()
state_agent <- make_state(env_agent$t, env_agent$soc)
agent_equity_curve <- c(env_agent$equity)

for (step in 1:max_steps_ep) {
  action <- which.max(predict(qnet_ddqn_eval, matrix(state_agent, nrow = 1), verbose = 0)) - 1L
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
max_len <- max(length(agent_equity_curve), length(random_equity_curve), length(heuristic_equity_curve), length(baseline_equity_curve))

plot_df <- data.frame(
  Step = 1:max_len,
  DDQN_Agent = c(agent_equity_curve, rep(NA, max_len - length(agent_equity_curve))),
  Baseline_Agent = c(baseline_equity_curve, rep(NA, max_len - length(baseline_equity_curve))),
  Random = c(random_equity_curve, rep(NA, max_len - length(random_equity_curve))),
  Heuristic = c(heuristic_equity_curve, rep(NA, max_len - length(heuristic_equity_curve)))
)

library(ggplot2)

ggplot(plot_df, aes(x = Step)) +
  geom_line(aes(y = DDQN_Agent, colour = "DDQN Agent")) +
  geom_line(aes(y = Baseline_Agent, colour = "Baseline Agent")) +
  geom_line(aes(y = Random, colour = "Random Policy")) +
  geom_line(aes(y = Heuristic, colour = "Heuristic Policy")) +
  labs(title = "Battery Agent vs. Benchmarks", y = "Equity", x = "Time Steps") +
  scale_colour_manual("", values = c("DDQN Agent" = "blue", "Baseline Agent" = "purple", "Random Policy" = "red", "Heuristic Policy" = "green")) +
  theme_minimal()
