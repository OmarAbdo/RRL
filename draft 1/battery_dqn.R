# Battery Dispatch Reinforcement Learning
# Task 1: Baseline DQN Agent for Battery Dispatch

# Load required libraries
library(keras)
library(dplyr)
library(readr)
library(lubridate)

# Battery parameters
battery_capacity <- 1.0  # MWh
power_limit <- 0.5        # MW (charge/discharge)
roundtrip_efficiency <- 0.9
degradation_cost <- 20    # EUR per MWh throughput
time_interval <- 0.25     # 15 minutes in hours

# Load electricity price data
price_data <- read_csv("Task_4/data_task4.csv")
colnames(price_data) <- c("datetime", "price")
prices <- price_data$price

# HYPER-PARAMETERS
window_size <- 24  # 6 hours of 15-min intervals
state_dim <- window_size + 3  # normalized prices + SoC + hour + weekday
hidden_units <- 64
learning_rate <- 1e-3
gamma <- 0.99
episodes <- 50
batch_size <- 32
replay_capacity <- 5000
warmup_mem <- 1000
target_sync_freq <- 10
eps_start <- 1.0
eps_final <- 0.05

# Environment functions
make_state <- function(t, soc) {
  # Handle zero variance case in price normalization
  price_window <- prices[(t - window_size + 1):t]
  if (sd(price_window) < 1e-5) {
    normalized_prices <- rep(0, length(price_window))
  } else {
    normalized_prices <- scale(price_window)[,1]
  }
  
  # Extract time features
  current_time <- as.POSIXlt(price_data$datetime[t])
  hour <- current_time$hour / 24
  weekday <- current_time$wday / 7
  
  # State: normalized prices + SoC + time features
  c(normalized极prices, soc, hour, weekday)
}

env_reset <- function() {
  list(t = window_size, 
       soc = 0.5, 
       equity = 1.0, 
       done = FALSE)
}

env_step <- function(env, action) {
  t <- env$t
  soc <- env$soc
  equity <- env$equity
  current_price <- prices[t]
  
  energy_change <- 0
  reward <- 0
  degradation <- 0
  
  # Action: 0 = idle, 极1 = charge, 2 = discharge
  if (action == 1 && soc < battery_capacity) {
    # Charging: max charge limited by capacity and power limit
    max_charge <- min(battery_capacity - soc, power_limit * time_interval)
    energy_change <- max_charge * roundtrip_efficiency
    cost <- max_charge * current_price
    degradation <- max_charge * degradation_cost
    reward <- - (cost + degradation)
    soc <- soc + energy_change
  } 
  else if (action == 2 && soc > 0) {
    # Discharging: max discharge limited by current SOC and power limit
    max_discharge <- min(soc, power_limit * time_interval)
    energy_change <- max_discharge
    revenue <- max_discharge * current_price * roundtrip_efficiency
    degradation <- max_discharge * degradation_cost
    reward <- revenue - degradation
    soc <- soc - energy_change
  }
  
  # Update equity and time
  equity <- equity + reward
  t <- t + 1
  done <- (t >= length(prices) - 1)
  
  next_env <- list(t = t, soc = soc, equity = equity, done = done)
  next_state <- make_state(t, soc)
  
  list(next_env = next_env, reward = reward, obs = next_state, done = done)
}

# Q-Network definition
build_qnet <- function() {
  input <- layer_input(shape = state_dim)
  hidden <- input %>% 
    layer_dense(hidden_units, activation = "relu")
  output <- hidden %>% 
    layer_dense(3, activation = "linear")  # 3 actions: idle, charge, discharge
  keras_model(input, output) %>% 
    compile(optimizer = optimizer_adam(learning_rate), loss = "mse")
}

# Initialize networks and replay buffer
qnet <- build_qnet()
target_net <- build_qnet()
target_net$set_weights(qnet$get_weights())

replay <- new.env(parent = emptyenv())
replay$S  <- array(0, dim = c(replay_capacity, state_dim))
replay$A  <- integer(replay_capacity)
replay$R  <- numeric(replay_capacity)
replay$S2 <- array(0, dim = c(replay_capacity, state_dim))
replay$D  <- integer(replay_capacity)
replay$idx <- 1L
replay$full <- FALSE

store_transition <- function(s, a, r, s2, done) {
  i <- replay$idx
  replay$S[i, ]  <- s
  replay$A[i]    <- a
  replay$R[i]    <- r
  replay$S2[i, ] <- s2
  replay$D[i]    <- done
  replay$idx     <- i %% replay_capacity + 1L
  if (i == replay_capacity) replay$full <- TRUE
}

sample_batch <- function(size) {
  max_i <- if (replay$full) replay_capacity else (replay$idx - 1L)
  idx   <- sample.int(max_i, size)
  list(S  = replay$S[idx, , drop = FALSE],
       A  = replay$A[idx],
       R  = replay$R[idx],
       S2 = replay$S2[idx, , drop = FALSE],
       D  = replay$D[idx])
}

# Training loop
eps_decay_rate <- log(eps_start / eps_final) / (episodes - 1)

epsilon <- function(ep) {
  eps_start * exp(-eps_decay_rate * (ep-1))
}

for (ep in 1:episodes) {
  env   <- env_reset()
  state <- make_state(env$t, env$soc)
  eps   <- epsilon(ep)
  ep_reward <- 0
  
  for (step in 1:2000) {  # Max steps per episode
    # Choose action (epsilon-greedy)
    if (runif(1) < eps) {
      action <- sample(0:2, 1)  # Explore
    } else {
      action <- which.max(predict(qnet, matrix(state, nrow = 1), verbose = 0)) - 1
    }
    
    # Take action
    res <- env_step(env, action)
    store_transition(state, action, res$reward, res$obs, res$next_env$done)
    
    state <- res$obs
    env   <- res$next_env
    ep_reward <- ep_reward + res$reward
    
    if (env$done) break
    
    # Train with batch from replay buffer
    if (replay$full || replay$idx > warmup_mem) {
      batch <- sample_batch(batch_size)
      q_next <- predict(target_net, batch$S2, verbose = 0)
      q_max  <- apply(q_next, 1, max)
      y      <- batch$R + (1 - batch$D) * gamma * q_max
      
      # Update Q-values
      current_q <- predict(qnet, batch$S, verbose = 0)
      for (j in 1:batch_size) {
        current_q[j, batch$A[j] + 1] <- y[j]
      }
      
      qnet %>% train_on_batch(batch$S, current_q)
    }
  }
  
  # Update target network periodically
  if (ep %% target_sync_freq == 0) {
    target_net$set_weights(qnet$get_weights())
  }
  
  cat(sprintf("Episode %3d | ε = %.3f | steps = %3d | reward = %.4f | equity = %.3f | SOC = %.2f\n",
              ep, eps, step, ep_reward, env$equity, env$soc))
}

# Save trained model
save_model_hdf5(qnet, "battery_dqn_weights.h5")
