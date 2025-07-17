# Group D - Task 4: Reinforcement Learning for Battery Dispatch
# Author: Cline
# Date: 2025-07-17

# =============================================================================
# 0. SETUP AND LIBRARIES
# =============================================================================

# --- Dynamic Working Directory ---
# Set working directory to the script's location
if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
  this_file <- rstudioapi::getActiveDocumentContext()$path
  setwd(dirname(this_file))
} else {
  args <- commandArgs(trailingOnly = FALSE)
  this_file_arg <- args[grep("--file=", args)]
  if (length(this_file_arg) > 0) {
    this_file <- normalizePath(sub("--file=", "", this_file_arg))
    setwd(dirname(this_file))
  }
}


# --- Package Management ---
required_packages <- c("keras", "tensorflow", "quantmod", "dplyr", "ggplot2", "tibble", "progress")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "http://cran.us.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

suppressPackageStartupMessages({
  library(keras)
  library(tensorflow)
  library(quantmod)
  library(dplyr)
  library(ggplot2)
  library(tibble)
  library(progress)
})

set.seed(42)

# =============================================================================
# 1. HYPERPARAMETERS AND CONFIGURATION
# =============================================================================

# --- Environment Parameters ---
BATTERY_CAPACITY_MWH <- 1.0
POWER_LIMIT_MW <- 0.5
EFFICIENCY <- 0.9
INTERVAL_HOURS <- 0.25 # 15 minutes
DEGRADATION_COST_EUR_MWH <- 20
WINDOW_SIZE <- 10

# --- Training Hyperparameters ---
EPISODES <- 10 # Total number of training episodes
LEARNING_RATE <- 1e-3
GAMMA <- 0.99 # Discount factor for future rewards
REPLAY_CAPACITY <- 5000
WARMUP_MEM <- 1000
BATCH_SIZE <- 32
TARGET_SYNC_FREQ <- 10 # Sync target network every 10 episodes

# --- Epsilon-Greedy Policy Parameters ---
EPS_START <- 1.0
EPS_FINAL <- 0.05
EPS_DECAY_RATE <- log(EPS_START / EPS_FINAL) / (EPISODES - 1)

# --- Network Architecture ---
STATE_DIM <- WINDOW_SIZE + 1
ACTION_DIM <- 3 # (Hold, Charge, Discharge)
HIDDEN_UNITS <- 32


# =============================================================================
# 2. DATA LOADING AND PREPARATION
# =============================================================================

load_and_prepare_data <- function(file_path, train_split = 0.8) {
  data <- read.csv(file_path)
  data$DateTime.UTC. <- as.POSIXct(data$DateTime.UTC., format = "%Y-%m-%dT%H:%M:%SZ")
  data <- data[rev(1:nrow(data)), ]
  
  price <- data$Price.Currency.MWh.
  rets <- na.omit(diff(price) / price[-length(price)]) * 100
  
  train_size <- floor(train_split * length(rets))
  train_rets <- rets[1:train_size]
  test_rets <- rets[(train_size + 1):length(rets)]
  
  train_prices <- price[1:train_size]
  test_prices <- price[(train_size + 1):length(price)]
  
  list(
    train_returns = train_rets,
    test_returns = test_rets,
    train_prices = train_prices,
    test_prices = test_prices
  )
}

# =============================================================================
# 3. ENVIRONMENT DEFINITION
# =============================================================================

make_state <- function(t, soc, returns_data) {
  c(as.numeric(returns_data[(t - WINDOW_SIZE + 1):t]), soc)
}

env_reset <- function(prices_data, returns_data) {
  list(
    t = WINDOW_SIZE,
    soc = 0.5,
    equity = 0,
    done = FALSE,
    prices = prices_data,
    returns = returns_data
  )
}

env_step <- function(env, action) {
  t2 <- env$t + 1
  soc <- env$soc
  equity <- env$equity
  
  if (t2 >= length(env$prices)) {
    return(list(next_env = env, reward = 0, obs = rep(0, STATE_DIM)))
  }
  
  current_price <- env$prices[t2]
  reward <- 0
  energy_per_step <- POWER_LIMIT_MW * INTERVAL_HOURS
  
  if (action == 1) { # Charge
    charge_amount <- min(energy_per_step, BATTERY_CAPACITY_MWH - soc * BATTERY_CAPACITY_MWH)
    cost <- charge_amount * current_price
    degradation <- charge_amount * DEGRADATION_COST_EUR_MWH
    reward <- -cost - degradation
    soc_next <- soc + (charge_amount * EFFICIENCY) / BATTERY_CAPACITY_MWH
  } else if (action == 2) { # Discharge
    discharge_amount <- min(energy_per_step, soc * BATTERY_CAPACITY_MWH)
    revenue <- (discharge_amount * EFFICIENCY) * current_price
    degradation <- discharge_amount * DEGRADATION_COST_EUR_MWH
    reward <- revenue - degradation
    soc_next <- soc - discharge_amount / BATTERY_CAPACITY_MWH
  } else { # Hold
    soc_next <- soc
    reward <- 0
  }
  
  soc_next <- max(0, min(1, soc_next))
  equity_next <- equity + reward
  done <- (t2 >= length(env$prices) - 1)
  
  list(
    next_env = list(t = t2, soc = soc_next, equity = equity_next, done = done, prices = env$prices, returns = env$returns),
    reward = reward,
    obs = make_state(t2, soc_next, env$returns)
  )
}

# =============================================================================
# 4. DQN AGENT AND TRAINING
# =============================================================================

create_replay_buffer <- function(capacity, state_dim) {
  buffer <- new.env(parent = emptyenv())
  buffer$S <- array(0, dim = c(capacity, state_dim))
  buffer$A <- integer(capacity)
  buffer$R <- numeric(capacity)
  buffer$S2 <- array(0, dim = c(capacity, state_dim))
  buffer$D <- integer(capacity)
  buffer$idx <- 1L
  buffer$full <- FALSE
  buffer
}

store_transition <- function(buffer, s, a, r, s2, done) {
  i <- buffer$idx
  capacity <- nrow(buffer$S)
  buffer$S[i, ] <- s
  buffer$A[i] <- a
  buffer$R[i] <- r
  buffer$S2[i, ] <- s2
  buffer$D[i] <- done
  buffer$idx <- i %% capacity + 1L
  if (i == capacity) buffer$full <- TRUE
}

sample_batch <- function(buffer, size) {
  capacity <- nrow(buffer$S)
  max_i <- if (buffer$full) capacity else (buffer$idx - 1L)
  idx <- sample.int(max_i, size)
  list(
    S = buffer$S[idx, , drop = FALSE],
    A = buffer$A[idx],
    R = buffer$R[idx],
    S2 = buffer$S2[idx, , drop = FALSE],
    D = buffer$D[idx]
  )
}

build_qnet <- function() {
  input <- layer_input(shape = STATE_DIM)
  hidden <- input %>% layer_dense(HIDDEN_UNITS, activation = "relu")
  output <- hidden %>% layer_dense(ACTION_DIM, activation = "linear")
  
  model <- keras_model(input, output)
  model %>% compile(
    optimizer = tf$keras$optimizers$legacy$Adam(learning_rate = LEARNING_RATE),
    loss = "mse"
  )
  model
}

train_agent <- function(env_prices, env_returns, agent_type = "DQN") {
  qnet <- build_qnet()
  target_net <- build_qnet()
  target_net$set_weights(qnet$get_weights())
  replay <- create_replay_buffer(REPLAY_CAPACITY, STATE_DIM)
  
  cat(sprintf("\n--- Starting %s Agent Training ---\n", agent_type))
  pb_episodes <- progress_bar$new(
    format = "  Episode :ep/:total [:bar] :percent in :elapsed | ETA: :eta",
    total = EPISODES, clear = FALSE, width = 60
  )
  
  for (ep in 1:EPISODES) {
    env <- env_reset(env_prices, env_returns)
    state <- make_state(env$t, env$soc, env$returns)
    ep_reward <- 0
    max_steps <- length(env_returns) - WINDOW_SIZE - 1
    
    pb_steps <- progress_bar$new(
      format = "    Step :step/:total [:bar] :percent",
      total = max_steps, clear = FALSE, width = 50
    )
    
    for (step in 1:max_steps) {
      eps <- EPS_START * exp(-EPS_DECAY_RATE * (ep - 1))
      if (runif(1) < eps) {
        action <- sample(0:2, 1)
      } else {
        action <- which.max(predict(qnet, matrix(state, nrow = 1), verbose = 0)) - 1
      }
      
      res <- env_step(env, action)
      store_transition(replay, state, action, res$reward, res$obs, res$next_env$done)
      
      state <- res$obs
      env <- res$next_env
      ep_reward <- ep_reward + res$reward
      
      if (replay$full || replay$idx > WARMUP_MEM) {
        batch <- sample_batch(replay, BATCH_SIZE)
        
        if (agent_type == "DDQN") {
          q_online_next <- predict(qnet, batch$S2, verbose = 0)
          best_actions <- apply(q_online_next, 1, which.max) - 1
          q_target_next <- predict(target_net, batch$S2, verbose = 0)
          q_max <- q_target_next[cbind(seq_len(BATCH_SIZE), best_actions + 1)]
        } else {
          q_next <- predict(target_net, batch$S2, verbose = 0)
          q_max <- apply(q_next, 1, max)
        }
        
        y <- batch$R + (1 - batch$D) * GAMMA * q_max
        y_keras <- predict(qnet, batch$S, verbose = 0)
        y_keras[cbind(seq_len(BATCH_SIZE), batch$A + 1)] <- y
        qnet %>% train_on_batch(batch$S, y_keras)
      }
      
      pb_steps$tick()
      if (env$done) break
    }
    
    if (ep %% TARGET_SYNC_FREQ == 0) {
      target_net$set_weights(qnet$get_weights())
    }
    
    pb_episodes$tick()
    cat(sprintf(" | Ep. Reward: %.2f | Final Equity: %.2f\n", ep_reward, env$equity))
  }
  
  return(qnet)
}

# =============================================================================
# 5. SCRIPT EXECUTION
# =============================================================================

# --- Load Data ---
data <- load_and_prepare_data("data_task4.csv")

# --- T1: Baseline Agent ---
baseline_agent <- train_agent(data$train_prices, data$train_returns, agent_type = "DQN")
save_model_weights_hdf5(baseline_agent, "model_weights_baseline.h5")
cat("\nBaseline agent weights saved to model_weights_baseline.h5\n")

# --- T4: DDQN Enhancement ---
ddqn_agent <- train_agent(data$train_prices, data$train_returns, agent_type = "DDQN")
save_model_weights_hdf5(ddqn_agent, "model_weights_ddqn.h5")
cat("DDQN agent weights saved to model_weights_ddqn.h5\n")

# --- Evaluation Function ---
run_evaluation <- function(agent, prices, returns) {
  env <- env_reset(prices, returns)
  state <- make_state(env$t, env$soc, env$returns)
  equity_curve <- c(env$equity)
  
  max_steps <- length(returns) - WINDOW_SIZE - 1
  for (step in 1:max_steps) {
    action <- if (is.null(agent)) {
      sample(0:2, 1)
    } else if (is.character(agent) && agent == "heuristic") {
      current_price <- prices[env$t + 1]
      if (current_price < 0) 1 else if (current_price > 0) 2 else 0
    } else {
      which.max(predict(agent, matrix(state, nrow = 1), verbose = 0)) - 1
    }
    
    res <- env_step(env, action)
    state <- res$obs
    env <- res$next_env
    equity_curve <- c(equity_curve, env$equity)
    if (env$done) break
  }
  equity_curve
}

# --- T2 & T3: Benchmarks and Evaluation ---
cat("\n--- Evaluating Agents and Benchmarks ---\n")

baseline_model <- build_qnet()
load_model_weights_hdf5(baseline_model, "model_weights_baseline.h5")

ddqn_model <- build_qnet()
load_model_weights_hdf5(ddqn_model, "model_weights_ddqn.h5")

baseline_equity <- run_evaluation(baseline_model, data$test_prices, data$test_returns)
ddqn_equity <- run_evaluation(ddqn_model, data$test_prices, data$test_returns)
random_equity <- run_evaluation(NULL, data$test_prices, data$test_returns)
heuristic_equity <- run_evaluation("heuristic", data$test_prices, data$test_returns)

max_len <- max(length(baseline_equity), length(ddqn_equity), length(random_equity), length(heuristic_equity))
plot_df <- data.frame(
  Step = 1:max_len,
  Baseline = c(baseline_equity, rep(NA, max_len - length(baseline_equity))),
  DDQN = c(ddqn_equity, rep(NA, max_len - length(ddqn_equity))),
  Random = c(random_equity, rep(NA, max_len - length(random_equity))),
  Heuristic = c(heuristic_equity, rep(NA, max_len - length(heuristic_equity)))
)

ggplot(plot_df, aes(x = Step)) +
  geom_line(aes(y = Baseline, colour = "Baseline DQN")) +
  geom_line(aes(y = DDQN, colour = "DDQN")) +
  geom_line(aes(y = Random, colour = "Random Policy")) +
  geom_line(aes(y = Heuristic, colour = "Heuristic Policy")) +
  labs(title = "Agent Performance vs. Benchmarks on Test Data", y = "Cumulative Equity (EUR)", x = "Time Steps (15 min intervals)") +
  scale_colour_manual("", values = c("Baseline DQN" = "blue", "DDQN" = "red", "Random Policy" = "green", "Heuristic Policy" = "purple")) +
  theme_minimal()

cat("\nEvaluation complete. Plot generated.\n")
