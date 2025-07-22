# Group D - Task 4: Reinforcement Learning Battery Trading

# --- Dynamic Working Directory Resolution ---
if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
  this_file <- rstudioapi::getActiveDocumentContext()$path
} else {
  args <- commandArgs(trailingOnly = FALSE)
  this_file <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
}

setwd(dirname(this_file))

# === CONFIGURATION ===
# Set training mode: "debug" for quick testing, "full" for complete training
TRAINING_MODE <- "debug" # Change to "full" for actual training

# Configuration based on training mode
if (TRAINING_MODE == "debug") {
  CONFIG <- list(
    # Data parameters
    train_days = 30, # 30 days for debugging
    test_days = 7, # 7 days for testing
    window_size = 24, # 6 hours lookback (24 * 15min = 6h)

    # Training parameters
    episodes = 5, # Few episodes for quick testing
    max_steps_per_episode = 1000,

    # DQN parameters
    learning_rate = 0.01, # Higher LR for faster learning
    batch_size = 32,
    replay_buffer_size = 5000,
    target_update_freq = 100,
    epsilon_start = 1.0,
    epsilon_end = 0.1,
    epsilon_decay_steps = 2000,
    warmup_steps = 500
  )
} else {
  CONFIG <- list(
    # Data parameters
    train_days = 365, # 1 year of training data
    test_days = 90, # 3 months of testing
    window_size = 96, # 24 hours lookback

    # Training parameters
    episodes = 50,
    max_steps_per_episode = 10000,

    # DQN parameters
    learning_rate = 0.001,
    batch_size = 64,
    replay_buffer_size = 50000,
    target_update_freq = 1000,
    epsilon_start = 1.0,
    epsilon_end = 0.01,
    epsilon_decay_steps = 20000,
    warmup_steps = 5000
  )
}

# Battery specifications 
BATTERY <- list(
  capacity_mwh = 1.0, # 1 MWh capacity
  power_mw = 0.5, # 0.5 MW power limit
  efficiency = 0.9, # 90% round-trip efficiency
  degradation_cost = 20, # EUR/MWh throughput cost
  timestep_hours = 0.25 # 15 minutes = 0.25 hours
)

# === SETUP ===
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(keras)
  library(tensorflow)
})

set.seed(42)
tf$random$set_seed(42)

# === DATA LOADING AND PREPROCESSING ===
load_and_prepare_data <- function() {
  cat("Loading data...\n")

  # Load data
  data <- read_csv("data_task4.csv", show_col_types = FALSE)

  # Parse datetime and sort chronologically
  data$datetime <- as.POSIXct(data$`DateTime(UTC)`, format = "%Y-%m-%dT%H:%M:%SZ")
  data <- data %>%
    arrange(datetime) %>%
    select(datetime, price = `Price[Currency/MWh]`)

  # Calculate data splits
  steps_per_day <- 96 # 24 hours * 4 (15-min intervals)
  train_steps <- CONFIG$train_days * steps_per_day
  test_steps <- CONFIG$test_days * steps_per_day

  # Extract prices
  total_needed <- train_steps + test_steps
  if (nrow(data) < total_needed) {
    stop(sprintf("Not enough data. Need %d steps, have %d", total_needed, nrow(data)))
  }

  train_prices <- data$price[1:train_steps]
  test_prices <- data$price[(train_steps + 1):(train_steps + test_steps)]

  cat(sprintf(
    "Data loaded: %d training steps, %d test steps\n",
    length(train_prices), length(test_prices)
  ))

  return(list(train = train_prices, test = test_prices))
}

# === ENVIRONMENT ===
create_battery_environment <- function(prices) {
  reset_environment <- function() {
    list(
      step = CONFIG$window_size + 1, # Start after window
      soc = 0.5, # Start at 50% charge
      total_profit = 0,
      done = FALSE
    )
  }

  get_state <- function(env) {
    # Current position in price series
    current_step <- env$step

    # Price window (last window_size prices)
    price_window <- prices[(current_step - CONFIG$window_size):(current_step - 1)]

    # Normalize prices (simple z-score normalization)
    price_mean <- mean(price_window)
    price_sd <- sd(price_window)
    if (price_sd > 0) {
      normalized_prices <- (price_window - price_mean) / price_sd
    } else {
      normalized_prices <- rep(0, length(price_window))
    }

    # Current price relative to window
    current_price <- prices[current_step]
    current_price_norm <- if (price_sd > 0) (current_price - price_mean) / price_sd else 0

    # State vector: [normalized_price_window, current_price_norm, soc]
    state <- c(normalized_prices, current_price_norm, env$soc)

    return(state)
  }

  step_environment <- function(env, action) {
    current_price <- prices[env$step]

    # Calculate energy change based on action
    # Action: 0 = hold, 1 = charge, 2 = discharge
    energy_change <- 0
    power_used <- 0

    if (action == 1) { 
      # Charge
      max_charge <- BATTERY$power_mw * BATTERY$timestep_hours
      # Don't exceed capacity
      available_capacity <- BATTERY$capacity_mwh - env$soc
      energy_change <- min(max_charge, available_capacity)
      power_used <- energy_change / BATTERY$efficiency # Account for losses
    } else if (action == 2) { 
      # Discharge
      max_discharge <- BATTERY$power_mw * BATTERY$timestep_hours
      # Don't go below 0
      available_energy <- env$soc
      energy_change <- -min(max_discharge, available_energy)
      power_used <- abs(energy_change) * BATTERY$efficiency # Account for losses
    }

    # Calculate costs and revenues
    if (action == 1) { # Charging - we buy electricity
      electricity_cost <- power_used * current_price
      degradation_cost <- power_used * BATTERY$degradation_cost
      reward <- -(electricity_cost + degradation_cost)
    } else if (action == 2) { # Discharging - we sell electricity
      electricity_revenue <- power_used * current_price
      degradation_cost <- abs(energy_change) * BATTERY$degradation_cost
      reward <- electricity_revenue - degradation_cost
    } else { # Holding
      reward <- 0
    }

    # Update environment
    new_env <- list(
      step = env$step + 1,
      soc = env$soc + energy_change,
      total_profit = env$total_profit + reward,
      done = env$step + 1 >= length(prices)
    )

    return(list(env = new_env, reward = reward, state = get_state(new_env)))
  }

  return(list(reset = reset_environment, step = step_environment, get_state = get_state))
}

# === NEURAL NETWORK ===
build_dqn <- function(state_size, action_size = 3, hidden_units = 64) {
  model <- keras_model_sequential() %>%
    layer_dense(units = hidden_units, activation = "relu", input_shape = c(state_size)) %>%
    layer_dense(units = hidden_units, activation = "relu") %>%
    layer_dense(units = action_size, activation = "linear")

  model %>% compile(
    optimizer = optimizer_adam(learning_rate = CONFIG$learning_rate),
    loss = "mse"
  )

  return(model)
}

# === REPLAY BUFFER ===
create_replay_buffer <- function(max_size, state_size) {
  buffer <- list(
    states = array(0, c(max_size, state_size)),
    actions = integer(max_size),
    rewards = numeric(max_size),
    next_states = array(0, c(max_size, state_size)),
    dones = logical(max_size),
    size = 0,
    index = 1
  )

  add_experience <- function(state, action, reward, next_state, done) {
    buffer$states[buffer$index, ] <<- state
    buffer$actions[buffer$index] <<- action
    buffer$rewards[buffer$index] <<- reward
    buffer$next_states[buffer$index, ] <<- next_state
    buffer$dones[buffer$index] <<- done

    buffer$index <<- buffer$index %% max_size + 1
    if (buffer$size < max_size) buffer$size <<- buffer$size + 1
  }

  sample_batch <- function(batch_size) {
    if (buffer$size < batch_size) {
      return(NULL)
    }

    indices <- sample(1:buffer$size, batch_size)

    list(
      states = buffer$states[indices, , drop = FALSE],
      actions = buffer$actions[indices],
      rewards = buffer$rewards[indices],
      next_states = buffer$next_states[indices, , drop = FALSE],
      dones = buffer$dones[indices]
    )
  }

  return(list(add = add_experience, sample = sample_batch, size = function() buffer$size))
}

# === TRAINING FUNCTION ===
train_dqn <- function(prices, use_double_dqn = FALSE) {
  cat(sprintf(
    "Training %s on %d price points...\n",
    if (use_double_dqn) "Double DQN" else "DQN", length(prices)
  ))

  # Initialize environment and networks
  env_factory <- create_battery_environment(prices)
  state_size <- CONFIG$window_size + 2 # price window + current price + soc

  online_net <- build_dqn(state_size)
  target_net <- build_dqn(state_size)
  target_net %>% set_weights(get_weights(online_net))

  replay_buffer <- create_replay_buffer(CONFIG$replay_buffer_size, state_size)

  # Training metrics
  episode_profits <- numeric(CONFIG$episodes)
  training_step <- 0

  for (episode in 1:CONFIG$episodes) {
    env <- env_factory$reset()
    state <- env_factory$get_state(env)
    episode_profit <- 0
    step_count <- 0

    while (!env$done && step_count < CONFIG$max_steps_per_episode) {
      # Epsilon-greedy action selection
      epsilon <- CONFIG$epsilon_end + (CONFIG$epsilon_start - CONFIG$epsilon_end) *
        exp(-training_step / CONFIG$epsilon_decay_steps)

      if (runif(1) < epsilon) {
        action <- sample(0:2, 1)
      } else {
        q_values <- predict(online_net, array(state, c(1, length(state))), verbose = 0)
        action <- which.max(q_values[1, ]) - 1
      }

      # Take action
      result <- env_factory$step(env, action)

      # Store experience
      replay_buffer$add(state, action, result$reward, result$state, result$env$done)

      # Training
      if (replay_buffer$size() >= CONFIG$warmup_steps) {
        batch <- replay_buffer$sample(CONFIG$batch_size)
        if (!is.null(batch)) {
          if (use_double_dqn) {
            # Double DQN update
            next_q_online <- predict(online_net, batch$next_states, verbose = 0)
            next_actions <- apply(next_q_online, 1, which.max)
            next_q_target <- predict(target_net, batch$next_states, verbose = 0)
            next_q_values <- sapply(1:nrow(batch$next_states), function(i) {
              next_q_target[i, next_actions[i]]
            })
          } else {
            # Standard DQN update
            next_q_values <- apply(predict(target_net, batch$next_states, verbose = 0), 1, max)
          }

          targets <- batch$rewards + 0.99 * next_q_values * (!batch$dones)

          current_q <- predict(online_net, batch$states, verbose = 0)
          for (i in 1:length(batch$actions)) {
            current_q[i, batch$actions[i] + 1] <- targets[i]
          }

          train_on_batch(online_net, batch$states, current_q)
        }
      }

      # Update target network
      if (training_step %% CONFIG$target_update_freq == 0) {
        target_net %>% set_weights(get_weights(online_net))
      }

      # Update for next iteration
      state <- result$state
      env <- result$env
      episode_profit <- env$total_profit
      step_count <- step_count + 1
      training_step <- training_step + 1
    }

    episode_profits[episode] <- episode_profit
    cat(sprintf(
      "Episode %d: Profit = %.2f EUR, Steps = %d, Epsilon = %.3f\n",
      episode, episode_profit, step_count, epsilon
    ))
  }

  return(list(model = online_net, profits = episode_profits))
}

# === EVALUATION FUNCTIONS ===
evaluate_policy <- function(policy_func, prices, policy_name) {
  cat(sprintf("Evaluating %s...\n", policy_name))

  env_factory <- create_battery_environment(prices)
  env <- env_factory$reset()

  profits <- c(env$total_profit)
  actions <- c()
  socs <- c(env$soc)

  while (!env$done) {
    state <- env_factory$get_state(env)
    action <- policy_func(state, env)
    result <- env_factory$step(env, action)

    profits <- c(profits, result$env$total_profit)
    actions <- c(actions, action)
    socs <- c(socs, result$env$soc)

    env <- result$env
  }

  cat(sprintf("%s final profit: %.2f EUR\n", policy_name, tail(profits, 1)))
  return(list(profits = profits, actions = actions, socs = socs))
}

# Policy functions
random_policy <- function(state, env) sample(0:2, 1)

heuristic_policy <- function(state, env) {
  current_price <- tail(state, 2)[1] # Current normalized price

  # Simple thresholds based on normalized price
  if (current_price < -0.5 && env$soc < 0.9) {
    return(1)
  } # Charge when price low
  if (current_price > 0.5 && env$soc > 0.1) {
    return(2)
  } # Discharge when price high
  return(0) # Hold otherwise
}

dqn_policy <- function(model) {
  function(state, env) {
    q_values <- predict(model, array(state, c(1, length(state))), verbose = 0)
    which.max(q_values[1, ]) - 1
  }
}

# === MAIN EXECUTION ===
main <- function() {
  cat("=== Battery Trading RL Agent ===\n")
  cat(sprintf("Running in %s mode\n", TRAINING_MODE))

  # Load data
  data <- load_and_prepare_data()

  # T1: Baseline DQN Agent
  cat("\n=== T1: Training Baseline DQN ===\n")
  dqn_result <- train_dqn(data$train, use_double_dqn = FALSE)

  # T4: Double DQN Enhancement
  cat("\n=== T4: Training Double DQN ===\n")
  ddqn_result <- train_dqn(data$train, use_double_dqn = TRUE)

  # T5: Save model weights
  cat("\n=== T5: Saving Model Weights ===\n")
  save_model_weights_hdf5(dqn_result$model, "model_weights_dqn.h5")
  save_model_weights_hdf5(ddqn_result$model, "model_weights_ddqn.h5")
  cat("Model weights saved successfully.\n")

  # T2 & T3: Evaluation and Benchmarking
  cat("\n=== T2 & T3: Evaluation and Benchmarking ===\n")

  # Evaluate all policies
  results <- list(
    dqn = evaluate_policy(dqn_policy(dqn_result$model), data$test, "DQN"),
    ddqn = evaluate_policy(dqn_policy(ddqn_result$model), data$test, "Double DQN"),
    random = evaluate_policy(random_policy, data$test, "Random"),
    heuristic = evaluate_policy(heuristic_policy, data$test, "Heuristic")
  )

  # Create comparison plots
  cat("\nCreating evaluation plots...\n")

  # Plot training progress
  training_df <- data.frame(
    Episode = 1:CONFIG$episodes,
    DQN = dqn_result$profits,
    DDQN = ddqn_result$profits
  )

  training_plot <- ggplot(training_df) +
    geom_line(aes(x = Episode, y = DQN, color = "DQN"), size = 1) +
    geom_line(aes(x = Episode, y = DDQN, color = "Double DQN"), size = 1) +
    labs(title = "Training Progress", x = "Episode", y = "Profit (EUR)", color = "Method") +
    theme_minimal()

  ggsave("training_progress.png", training_plot, width = 10, height = 6)

  # Plot evaluation results
  max_steps <- min(
    length(results$dqn$profits), length(results$ddqn$profits),
    length(results$random$profits), length(results$heuristic$profits)
  )

  eval_df <- data.frame(
    Step = 1:max_steps,
    DQN = results$dqn$profits[1:max_steps],
    DDQN = results$ddqn$profits[1:max_steps],
    Random = results$random$profits[1:max_steps],
    Heuristic = results$heuristic$profits[1:max_steps]
  )

  eval_long <- tidyr::pivot_longer(eval_df, cols = -Step, names_to = "Policy", values_to = "Profit")

  eval_plot <- ggplot(eval_long, aes(x = Step, y = Profit, color = Policy)) +
    geom_line(size = 1) +
    labs(
      title = "Policy Comparison on Test Set",
      x = "Time Step", y = "Cumulative Profit (EUR)"
    ) +
    theme_minimal()

  ggsave("policy_comparison.png", eval_plot, width = 12, height = 6)

  # Summary results
  final_profits <- sapply(results, function(x) tail(x$profits, 1))
  cat("\n=== FINAL RESULTS ===\n")
  for (policy in names(final_profits)) {
    cat(sprintf("%s: %.2f EUR\n", toupper(policy), final_profits[policy]))
  }

  cat(sprintf("\nTraining completed in %s mode\n", TRAINING_MODE))
  if (TRAINING_MODE == "debug") {
    cat("To run full training, change TRAINING_MODE to 'full' at the top of the script\n")
  }
}

# Run the main function
main()
