# Group D - Task 4

# --- Dynamic Working Directory Resolution ---
if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
  this_file <- rstudioapi::getActiveDocumentContext()$path
} else {
  args <- commandArgs(trailingOnly = FALSE)
  this_file <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
}

setwd(dirname(this_file))

# --- Initial Setup ---
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(keras)
  library(tibble)
  library(progress)
  library(tidyr)
})

set.seed(42)

# --- Load and Prepare Data ---
# According to task4.pdf, we will use the Austrian 15-minute day-ahead electricity prices.
# The data is in 'data_task4.csv'.
data <- read_csv("data_task4.csv")

# The data is in reverse chronological order, so we need to reverse it.
data <- data %>% arrange(`DateTime(UTC)`)

# We will use the price column for our environment.
prices <- data$`Price[Currency/MWh]`

# --- Configuration and Hyperparameters ---
# Data split
train_years <- 2
test_years <- 0.25
steps_per_year <- 365 * 24 * 4
train_end_idx <- steps_per_year * train_years
test_end_idx <- train_end_idx + (steps_per_year * test_years)
train_prices <- prices[1:train_end_idx]
test_prices <- prices[(train_end_idx + 1):test_end_idx]

# Environment
window_size <- 96 # 24 hours of 15-min intervals
start_soc <- 0.5 # Start at 50% SoC

# Battery Parameters
power_limit <- 0.5 # MW
usable_energy <- 1 # MWh
efficiency <- 0.9 # 90%
degradation_cost_per_mwh <- 20 # EUR/MWh

# DQN Hyperparameters
gamma <- 0.99 # Discount factor. High value for long-term profit maximization.
learning_rate <- 0.001 # Adam optimizer learning rate. Standard value.
replay_capacity <- 50000 # Size of replay buffer.
warmup_size <- 10000 # Steps before training, to populate the buffer.
max_steps_ep <- 96 * 182 # Max steps per episode (e.g., 6 months).
epsilon_start <- 1.0 # Start with full exploration.
epsilon_final <- 0.01 # End with almost full exploitation.
epsilon_decay_rate <- 0.0001 # Decay rate for epsilon.
batch_size <- 64 # Batch size for training.
total_episodes <- 50 # Total number of training episodes.
target_net_update_freq <- 1000 # update target network every 1000 steps.


# --- T1: Baseline Agent ---

# --- Environment Functions ---
# The state will consist of a window of past prices and the current state-of-charge (SoC).
make_state <- function(t, soc, prices_data) {
  # Ensure we don't go before the start of the price data
  start_index <- max(1, t - window_size + 1)
  price_window <- prices_data[start_index:t]

  # Pad with initial price if the window is not full
  if (length(price_window) < window_size) {
    padding <- rep(prices_data[1], window_size - length(price_window))
    price_window <- c(padding, price_window)
  }

  # Time-based features
  hour_of_day <- (t %% 96) / 96
  day_of_week <- (t %% (96 * 7)) / (96 * 7)
  day_of_year <- (t %% (96 * 365)) / (96 * 365)

  # Price statistics
  rolling_mean <- mean(price_window)
  rolling_sd <- sd(price_window)

  c(
    price_window, soc,
    sin(2 * pi * hour_of_day), cos(2 * pi * hour_of_day),
    sin(2 * pi * day_of_week), cos(2 * pi * day_of_week),
    sin(2 * pi * day_of_year), cos(2 * pi * day_of_year),
    rolling_mean, rolling_sd
  )
}

# Reset the environment to its initial state
env_reset <- function(prices_data) {
  list(
    t = window_size,
    soc = start_soc,
    profit = 0,
    done = FALSE,
    prices_data = prices_data
  )
}

# Execute a step in the environment
env_step <- function(env, action) {
  t <- env$t
  soc <- env$soc
  profit <- env$profit
  prices_data <- env$prices_data

  # Check if the episode is over
  if (t >= length(prices_data) - 1) {
    return(list(next_env = env, reward = 0, obs = rep(0, window_size + 1 + 8), done = TRUE))
  }

  current_price <- prices_data[t]

  # Action: 0 = do nothing, 1 = charge, 2 = discharge
  energy_change <- 0
  cost_or_revenue <- 0

  if (action == 1) { # Charge
    energy_change <- power_limit * 0.25 # 0.5 MW for 15 mins (0.25h)
    # Ensure we don't exceed usable energy capacity
    if (soc + energy_change > usable_energy) {
      energy_change <- usable_energy - soc
    }
    cost_or_revenue <- -(energy_change / efficiency) * current_price
    degradation_cost <- (energy_change / efficiency) * degradation_cost_per_mwh
    soc <- soc + energy_change
  } else if (action == 2) { # Discharge
    energy_change <- power_limit * 0.25
    # Ensure we don't go below 0 SoC
    if (soc - energy_change < 0) {
      energy_change <- soc
    }
    cost_or_revenue <- (energy_change * efficiency) * current_price
    degradation_cost <- (energy_change * efficiency) * degradation_cost_per_mwh
    soc <- soc - energy_change
  } else { # Do nothing
    degradation_cost <- 0
  }

  reward <- cost_or_revenue - degradation_cost
  profit <- profit + reward

  t <- t + 1
  done <- (t >= length(prices_data) - 1)

  next_env <- list(t = t, soc = soc, profit = profit, done = done, prices_data = prices_data)
  obs <- make_state(t, soc, prices_data)

  list(next_env = next_env, reward = reward, obs = obs, done = done)
}

# --- Replay Buffer ---
replay_buffer <- new.env()
state_size <- window_size + 1 + 8

replay_buffer$s <- matrix(0, nrow = replay_capacity, ncol = state_size)
replay_buffer$a <- integer(replay_capacity)
replay_buffer$r <- numeric(replay_capacity)
replay_buffer$s2 <- matrix(0, nrow = replay_capacity, ncol = state_size)
replay_buffer$done <- logical(replay_capacity)
replay_buffer$idx <- 1
replay_buffer$is_full <- FALSE

store_transition <- function(buffer, s, a, r, s2, done) {
  idx <- buffer$idx
  buffer$s[idx, ] <- s
  buffer$a[idx] <- a
  buffer$r[idx] <- r
  buffer$s2[idx, ] <- s2
  buffer$done[idx] <- done

  buffer$idx <- buffer$idx + 1
  if (buffer$idx > replay_capacity) {
    buffer$idx <- 1
    buffer$is_full <- TRUE
  }
}

sample_batch <- function(buffer, n) {
  max_index <- if (buffer$is_full) replay_capacity else (buffer$idx - 1)
  sampled_idx <- sample(1:max_index, size = n, replace = FALSE)

  list(
    s = buffer$s[sampled_idx, , drop = FALSE],
    a = buffer$a[sampled_idx],
    r = buffer$r[sampled_idx],
    s2 = buffer$s2[sampled_idx, , drop = FALSE],
    done = buffer$done[sampled_idx]
  )
}

# --- Network and Epsilon-Greedy ---
build_qnet <- function(input_nodes = state_size, output_nodes = 3, hidden_units = 32) {
  model <- keras_model_sequential() %>%
    layer_dense(units = hidden_units, activation = "relu", input_shape = c(input_nodes)) %>%
    layer_dense(units = output_nodes, activation = "linear")
  return(model)
}

epsilon_decay <- function(ep, start = epsilon_start, final = epsilon_final, decay_rate = epsilon_decay_rate) {
  final + (start - final) * exp(-decay_rate * ep)
}

# --- Training Loop ---

# Initialize networks
online_net <- build_qnet()
target_net <- build_qnet()
target_net %>% set_weights(online_net %>% get_weights())

online_net %>% compile(
  optimizer = optimizer_adam(learning_rate = learning_rate),
  loss = "mse"
)

episode_log <- list()

pb_episodes <- progress_bar$new(
  format = "Episodes [:bar] :percent eta: :eta",
  total = total_episodes, clear = FALSE, width = 60
)

for (ep in 1:total_episodes) {
  env <- env_reset(train_prices)
  state <- make_state(env$t, env$soc, env$prices_data)

  episode_profit <- 0

  pb_steps <- progress_bar$new(
    format = paste0("  Episode ", ep, " [:bar] :percent eta: :eta"),
    total = max_steps_ep, clear = FALSE, width = 60
  )

  for (step in 1:max_steps_ep) {
    pb_steps$tick()
    epsilon <- epsilon_decay(ep, start = epsilon_start, final = epsilon_final, decay_rate = epsilon_decay_rate)
    if (runif(1) < epsilon) {
      action <- sample(0:2, 1) # Explore
    } else {
      q_values <- online_net %>% predict(t(state), verbose = 0)
      action <- which.max(q_values) - 1 # Exploit (0, 1, or 2)
    }

    step_result <- env_step(env, action)
    next_state <- step_result$obs
    reward <- step_result$reward
    done <- step_result$done

    store_transition(replay_buffer, state, action, reward, next_state, done)

    state <- next_state
    env <- step_result$next_env
    episode_profit <- env$profit

    if (replay_buffer$idx > warmup_size || replay_buffer$is_full) {
      batch <- sample_batch(replay_buffer, batch_size)

      q_next <- target_net %>% predict(batch$s2, verbose = 0)
      q_next_max <- apply(q_next, 1, max)

      target_q <- batch$r + gamma * q_next_max * (1 - batch$done)

      current_q <- online_net %>% predict(batch$s, verbose = 0)

      for (i in 1:batch_size) {
        current_q[i, batch$a[i] + 1] <- target_q[i]
      }

      train_on_batch(online_net, batch$s, current_q)
    }

    if (step %% target_net_update_freq == 0 && (replay_buffer$idx > warmup_size || replay_buffer$is_full)) {
      target_net %>% set_weights(online_net %>% get_weights())
    }

    if (done) break
  }

  episode_log[[ep]] <- list(profit = episode_profit)
  pb_episodes$tick()
  cat(sprintf("\nEpisode: %d, Final Profit: %.2f\n", ep, episode_profit))
}

# --- T4: Mandatory Enhancement (Double-DQN) ---
# The core difference in DDQN is in the calculation of the target Q-value.
# Instead of using the target network to both select and evaluate the best next action,
# we use the online network to select the action and the target network to evaluate it.

# We will re-train a new agent with the DDQN update rule.
online_net_ddqn <- build_qnet()
target_net_ddqn <- build_qnet()
target_net_ddqn %>% set_weights(online_net_ddqn %>% get_weights())

online_net_ddqn %>% compile(
  optimizer = optimizer_adam(learning_rate = learning_rate),
  loss = "mse"
)

episode_log_ddqn <- list()
replay_buffer_ddqn <- new.env() # Use a new buffer for the DDQN agent
replay_buffer_ddqn$s <- matrix(0, nrow = replay_capacity, ncol = state_size)
replay_buffer_ddqn$a <- integer(replay_capacity)
replay_buffer_ddqn$r <- numeric(replay_capacity)
replay_buffer_ddqn$s2 <- matrix(0, nrow = replay_capacity, ncol = state_size)
replay_buffer_ddqn$done <- logical(replay_capacity)
replay_buffer_ddqn$idx <- 1
replay_buffer_ddqn$is_full <- FALSE

pb_episodes_ddqn <- progress_bar$new(
  format = "DDQN Episodes [:bar] :percent eta: :eta",
  total = total_episodes, clear = FALSE, width = 60
)

for (ep in 1:total_episodes) {
  env <- env_reset(train_prices)
  state <- make_state(env$t, env$soc, env$prices_data)

  episode_profit <- 0

  pb_steps_ddqn <- progress_bar$new(
    format = paste0("  DDQN Episode ", ep, " [:bar] :percent eta: :eta"),
    total = max_steps_ep, clear = FALSE, width = 60
  )

  for (step in 1:max_steps_ep) {
    pb_steps_ddqn$tick()
    epsilon <- epsilon_decay(ep, start = epsilon_start, final = epsilon_final, decay_rate = epsilon_decay_rate)
    if (runif(1) < epsilon) {
      action <- sample(0:2, 1)
    } else {
      q_values <- online_net_ddqn %>% predict(t(state), verbose = 0)
      action <- which.max(q_values) - 1
    }

    step_result <- env_step(env, action)
    next_state <- step_result$obs
    reward <- step_result$reward
    done <- step_result$done

    store_transition(replay_buffer_ddqn, state, action, reward, next_state, done)

    state <- next_state
    env <- step_result$next_env
    episode_profit <- env$profit

    if (replay_buffer_ddqn$idx > warmup_size || replay_buffer_ddqn$is_full) {
      batch <- sample_batch(replay_buffer_ddqn, batch_size)

      # DDQN update
      q_next_online <- online_net_ddqn %>% predict(batch$s2, verbose = 0)
      best_actions <- apply(q_next_online, 1, which.max)

      q_next_target <- target_net_ddqn %>% predict(batch$s2, verbose = 0)

      q_next_max <- sapply(1:batch_size, function(i) q_next_target[i, best_actions[i]])

      target_q <- batch$r + gamma * q_next_max * (1 - batch$done)

      current_q <- online_net_ddqn %>% predict(batch$s, verbose = 0)

      for (i in 1:batch_size) {
        current_q[i, batch$a[i] + 1] <- target_q[i]
      }

      train_on_batch(online_net_ddqn, batch$s, current_q)
    }

    if (step %% target_net_update_freq == 0 && (replay_buffer_ddqn$idx > warmup_size || replay_buffer_ddqn$is_full)) {
      target_net_ddqn %>% set_weights(online_net_ddqn %>% get_weights())
    }

    if (done) break
  }

  episode_log_ddqn[[ep]] <- list(profit = episode_profit)
  pb_episodes_ddqn$tick()
  cat(sprintf("\nDDQN Episode: %d, Final Profit: %.2f\n", ep, episode_profit))
}

# --- T5: Store your weights ---
save_model_weights_hdf5(online_net, "model_weights_baseline.h5")
save_model_weights_hdf5(online_net_ddqn, "model_weights_ddqn.h5")

cat("Model weights saved successfully.\n")

# --- T2 & T3: Enhanced Evaluation ---

# 1. Trained Agent Policy (DQN)
agent_policy <- function(state, env) {
  q_values <- online_net %>% predict(t(state), verbose = 0)
  which.max(q_values) - 1
}

# 2. DDQN Policy
ddqn_policy <- function(state, env) {
  q_values <- online_net_ddqn %>% predict(t(state), verbose = 0)
  which.max(q_values) - 1
}

# 3. Random Policy
random_policy <- function(state, env) {
  sample(0:2, 1)
}

# 4. Heuristic Policy
heuristic_policy <- function(state, env) {
  current_price <- env$prices_data[env$t]
  rolling_mean <- state[window_size + 8]
  rolling_sd <- state[window_size + 9]
  
  # Define thresholds based on rolling statistics (e.g., 1 standard deviation from the mean)
  charge_threshold <- rolling_mean - 1.0 * rolling_sd
  discharge_threshold <- rolling_mean + 1.0 * rolling_sd

  if (current_price < charge_threshold) {
    return(1) # Charge if the price is significantly low
  } else if (current_price > discharge_threshold) {
    return(2) # Discharge if the price is significantly high
  } else {
    return(0) # Do nothing
  }
}

# Function to evaluate a policy on the test set
evaluate_policy <- function(policy_func, prices_data, policy_name = "Policy") {
  env <- env_reset(prices_data)

  total_steps <- length(prices_data) - window_size
  if (total_steps <= 0) {
    cat(sprintf(
      "\nSkipping evaluation for %s: Not enough data points in test set (%d) for window size (%d).\n",
      policy_name, length(prices_data), window_size
    ))
    return(list(soc = c(env$soc), profit = c(env$profit), actions = c()))
  }

  state <- make_state(env$t, env$soc, env$prices_data)

  soc_history <- c(env$soc)
  profit_history <- c(env$profit)
  action_history <- c()

  pb_eval <- progress_bar$new(
    format = paste0("  Evaluating ", policy_name, " [:bar] :percent eta: :eta"),
    total = total_steps, clear = FALSE, width = 60
  )

  while (!env$done) {
    pb_eval$tick()
    action <- policy_func(state, env)

    step_result <- env_step(env, action)
    state <- step_result$obs
    env <- step_result$next_env

    soc_history <- c(soc_history, env$soc)
    profit_history <- c(profit_history, env$profit)
    action_history <- c(action_history, action)
  }

  return(list(soc = soc_history, profit = profit_history, actions = action_history))
}

# Run all evaluations
results_dqn <- evaluate_policy(agent_policy, test_prices, "DQN")
results_ddqn <- evaluate_policy(ddqn_policy, test_prices, "DDQN")
results_random <- evaluate_policy(random_policy, test_prices, "Random")
results_heuristic <- evaluate_policy(heuristic_policy, test_prices, "Heuristic")

# Check if evaluation produced results before proceeding
if (length(results_dqn$actions) > 0) {
  # Combine results into a single data frame
  n_steps <- length(results_dqn$actions)
  evaluation_df <- tibble(
    Step = 1:n_steps,
    Price = test_prices[window_size:(window_size + n_steps - 1)],
    DQN_Profit = results_dqn$profit[2:(n_steps + 1)],
    DQN_SoC = results_dqn$soc[2:(n_steps + 1)],
    DQN_Action = results_dqn$actions,
    DDQN_Profit = results_ddqn$profit[2:(n_steps + 1)],
    DDQN_SoC = results_ddqn$soc[2:(n_steps + 1)],
    DDQN_Action = results_ddqn$actions,
    Random_Profit = results_random$profit[2:(n_steps + 1)],
    Heuristic_Profit = results_heuristic$profit[2:(n_steps + 1)]
  )

  # Save evaluation results to CSV
  write_csv(evaluation_df, "evaluation_results.csv")

  # --- Plotting ---

  # 1. Cumulative Profit Plot
  profit_plot_df <- evaluation_df %>%
    select(Step, DQN_Profit, DDQN_Profit, Random_Profit, Heuristic_Profit) %>%
    rename(DQN = DQN_Profit, DDQN = DDQN_Profit, Random = Random_Profit, Heuristic = Heuristic_Profit) %>%
    tidyr::gather(key = "Policy", value = "Profit", -Step)

  profit_plot <- ggplot(profit_plot_df, aes(x = Step, y = Profit, color = Policy)) +
    geom_line(linewidth = 1) +
    labs(
      title = "Cumulative Profit Comparison",
      x = "Time Step (15-min interval)",
      y = "Cumulative Profit (EUR)",
      color = "Policy"
    ) +
    theme_minimal() +
    scale_color_brewer(palette = "Set1")

  ggsave("profit_plot.png", plot = profit_plot, width = 10, height = 6)

  # 2. Agent Behavior Plot
  behavior_df <- evaluation_df %>%
    select(Step, Price, DQN_SoC, DQN_Action, DDQN_SoC, DDQN_Action) %>%
    rename(SoC_DQN = DQN_SoC, Action_DQN = DQN_Action, SoC_DDQN = DDQN_SoC, Action_DDQN = DDQN_Action) %>%
    tidyr::gather(key = "Metric", value = "Value", -Step, -Price) %>%
    tidyr::separate(Metric, into = c("Metric", "Agent"), sep = "_") %>%
    tidyr::spread(key = Metric, value = Value)

  agent_behavior_plot <- ggplot(behavior_df, aes(x = Step)) +
    geom_line(aes(y = Price), color = "black", linewidth = 0.8) +
    geom_line(aes(y = SoC * max(behavior_df$Price, na.rm = TRUE)), color = "blue", linetype = "dashed") +
    geom_point(data = filter(behavior_df, Action == 1), aes(y = Price), color = "green", size = 2, alpha = 0.7) +
    geom_point(data = filter(behavior_df, Action == 2), aes(y = Price), color = "red", size = 2, alpha = 0.7) +
    scale_y_continuous(
      name = "Electricity Price (EUR/MWh)",
      sec.axis = sec_axis(~ . / max(behavior_df$Price, na.rm = TRUE), name = "State of Charge (SoC)")
    ) +
    facet_wrap(~Agent, nrow = 2) +
    labs(
      title = "Agent Trading Behavior vs. Market Price",
      x = "Time Step (15-min interval)"
    ) +
    theme_bw()

  ggsave("agent_behavior_plot.png", plot = agent_behavior_plot, width = 12, height = 8)

  cat("Evaluation complete. Plots and CSV saved.\n")
} else {
  cat("\nEvaluation and plotting skipped because the test set is too small for the given window size.\n")
}
