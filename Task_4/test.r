# Suppress TensorFlow C++ warnings
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

library(keras)
library(tensorflow)
library(ggplot2)
if (!requireNamespace("progress", quietly = TRUE)) install.packages("progress")
library(progress)

## CONFIGURATION AND HYPERPARAMETERS =======================================
# Battery physical constraints
BATTERY_CAPACITY_MWH <- 1.0
POWER_LIMIT_MW <- 0.5
ROUND_TRIP_EFFICIENCY <- 0.9
DEGRADATION_COST_EUR_PER_MWH <- 20
INTERVAL_HOURS <- 0.25
MAX_ENERGY_PER_STEP <- POWER_LIMIT_MW * INTERVAL_HOURS

# RL hyperparameters - improved values
LOOKBACK_WINDOW <- 20  # Increased for better pattern recognition
STATE_DIMENSION <- LOOKBACK_WINDOW + 3  # price window + SoC + current price + price trend
HIDDEN_UNITS <- 64  # Increased network capacity
TRAINING_EPISODES <- 10  # More training episodes
REPLAY_BUFFER_SIZE <- 10000  # Larger replay buffer
WARMUP_STEPS <- 1000  # More warmup steps
BATCH_SIZE <- 64  # Larger batch size
LEARNING_RATE <- 0.0001  # Lower learning rate for stability
GAMMA <- 0.99  # Discount factor
TARGET_UPDATE_FREQUENCY <- 20  # Less frequent target updates
EPSILON_START <- 1.0
EPSILON_END <- 0.01  # Lower final epsilon
EPSILON_DECAY_EPISODES <- TRAINING_EPISODES * 0.7  # Decay over 70% of training

# PER hyperparameters
PER_ALPHA <- 0.6
PER_BETA_START <- 0.4
PER_BETA_END <- 1.0
PER_EPSILON <- 1e-6

## DATA LOADING AND PREPROCESSING ==========================================
load_and_preprocess_data <- function(file_path) {
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  data$DateTime.UTC. <- as.POSIXct(data$DateTime.UTC., format = "%Y-%m-%dT%H:%M:%SZ")
  data <- data[order(data$DateTime.UTC.),]  # Ensure chronological order
  
  prices <- data$Price.Currency.MWh.
  
  # Feature engineering - normalize prices and create trend indicators
  price_normalized <- scale(prices)[,1]
  price_returns <- c(0, diff(prices) / prices[-length(prices)])
  price_trend <- c(0, diff(price_returns))
  
  # Handle any NaN or infinite values
  price_returns[is.infinite(price_returns) | is.na(price_returns)] <- 0
  price_trend[is.infinite(price_trend) | is.na(price_trend)] <- 0
  
  list(
    original_prices = prices,
    normalized_prices = price_normalized,
    returns = price_returns,
    trend = price_trend,
    timestamps = data$DateTime.UTC.
  )
}

## ENVIRONMENT IMPLEMENTATION ===============================================
BatteryEnvironment <- R6::R6Class("BatteryEnvironment",
  public = list(
    prices = NULL,
    returns = NULL,
    trend = NULL,
    current_step = NULL,
    max_steps = NULL,
    soc = NULL,
    equity = NULL,
    total_throughput = NULL,
    action_history = NULL,
    
    initialize = function(prices, returns, trend, start_step = LOOKBACK_WINDOW + 1) {
      self$prices <- prices
      self$returns <- returns
      self$trend <- trend
      self$current_step <- start_step
      self$max_steps <- length(prices) - 1
      self$reset()
    },
    
    reset = function() {
      self$current_step <- LOOKBACK_WINDOW + 1
      self$soc <- 0.5  # Start at 50% SoC
      self$equity <- 0.0  # Start with 0 equity
      self$total_throughput <- 0
      self$action_history <- c()
      return(self$get_state())
    },
    
    get_state = function() {
      if (self$current_step <= LOOKBACK_WINDOW) {
        stop("Current step is too early for state construction")
      }
      
      # Get price history window
      price_window <- self$returns[(self$current_step - LOOKBACK_WINDOW):(self$current_step - 1)]
      
      # Current market conditions
      current_price <- self$prices[self$current_step]
      current_trend <- self$trend[self$current_step]
      
      return(c(price_window, self$soc, current_price / 100, current_trend))
    },
    
    step = function(action) {
      if (self$current_step >= self$max_steps) {
        return(list(next_state = self$get_state(), reward = 0, done = TRUE))
      }
      
      current_price <- self$prices[self$current_step]
      reward <- 0
      energy_moved <- 0
      
      # Action execution with proper constraints
      if (action == 1) {  # Charge
        available_capacity <- BATTERY_CAPACITY_MWH - (self$soc * BATTERY_CAPACITY_MWH)
        charge_amount <- min(MAX_ENERGY_PER_STEP, available_capacity)
        
        if (charge_amount > 0) {
          # Account for charging efficiency
          energy_stored <- charge_amount * ROUND_TRIP_EFFICIENCY
          self$soc <- self$soc + (energy_stored / BATTERY_CAPACITY_MWH)
          
          # Calculate costs
          electricity_cost <- charge_amount * current_price
          degradation_cost <- charge_amount * DEGRADATION_COST_EUR_PER_MWH
          
          reward <- -(electricity_cost + degradation_cost)
          energy_moved <- charge_amount
        }
      } else if (action == 2) {  # Discharge
        available_energy <- self$soc * BATTERY_CAPACITY_MWH
        discharge_amount <- min(MAX_ENERGY_PER_STEP, available_energy)
        
        if (discharge_amount > 0) {
          # Account for discharging efficiency
          energy_delivered <- discharge_amount * ROUND_TRIP_EFFICIENCY
          self$soc <- self$soc - (discharge_amount / BATTERY_CAPACITY_MWH)
          
          # Calculate revenue
          electricity_revenue <- energy_delivered * current_price
          degradation_cost <- discharge_amount * DEGRADATION_COST_EUR_PER_MWH
          
          reward <- electricity_revenue - degradation_cost
          energy_moved <- discharge_amount
        }
      }
      # Action 0 (Hold) does nothing
      
      # Update tracking variables
      self$equity <- self$equity + reward
      self$total_throughput <- self$total_throughput + energy_moved
      self$action_history <- c(self$action_history, action)
      self$current_step <- self$current_step + 1
      
      # Ensure SoC remains within bounds
      self$soc <- max(0, min(1, self$soc))
      
      done <- self$current_step >= self$max_steps
      next_state <- if (!done) self$get_state() else self$get_state()
      
      return(list(
        next_state = next_state,
        reward = reward,
        done = done
      ))
    },
    
    is_done = function() {
      return(self$current_step >= self$max_steps)
    }
  )
)

## NEURAL NETWORK ARCHITECTURE =============================================
create_dqn_model <- function(state_dim, action_dim, hidden_units = HIDDEN_UNITS) {
  input_layer <- layer_input(shape = state_dim)
  
  # Enhanced architecture with batch normalization and dropout
  hidden1 <- input_layer %>%
    layer_dense(units = hidden_units, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.2)
  
  hidden2 <- hidden1 %>%
    layer_dense(units = hidden_units, activation = "relu") %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.2)
  
  # Value and advantage streams for dueling architecture
  value_stream <- hidden2 %>%
    layer_dense(units = hidden_units, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
  
  advantage_stream <- hidden2 %>%
    layer_dense(units = hidden_units, activation = "relu") %>%
    layer_dense(units = action_dim, activation = "linear")
  
  # Combine value and advantage streams
  # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
  output_layer <- layer_lambda(
    list(value_stream, advantage_stream),
    function(x) {
      value <- x[[1]]
      advantage <- x[[2]]
      advantage_mean <- k_mean(advantage, axis = 2, keepdims = TRUE)
      value + advantage - advantage_mean
    }
  )
  
  model <- keras_model(inputs = input_layer, outputs = output_layer)
  
  # Use Adam optimizer with gradient clipping
  optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate = LEARNING_RATE, clipnorm = 1.0)
  model %>% compile(optimizer = optimizer, loss = "mse")
  
  return(model)
}

## PRIORITIZED EXPERIENCE REPLAY ===========================================
PrioritizedReplayBuffer <- R6::R6Class("PrioritizedReplayBuffer",
  public = list(
    capacity = NULL,
    buffer = NULL,
    priorities = NULL,
    position = 0,
    size = 0,
    
    initialize = function(capacity) {
      self$capacity <- capacity
      self$buffer <- list()
      self$priorities <- numeric(capacity)
    },
    
    add = function(state, action, reward, next_state, done) {
      # Calculate max priority for new experience
      max_priority <- if (self$size > 0) max(self$priorities[1:self$size]) else 1.0
      
      idx <- (self$position %% self$capacity) + 1
      self$buffer[[idx]] <- list(
        state = state,
        action = action,
        reward = reward,
        next_state = next_state,
        done = done
      )
      self$priorities[idx] <- max_priority
      
      self$position <- self$position + 1
      self$size <- min(self$size + 1, self$capacity)
    },
    
    sample = function(batch_size, alpha = PER_ALPHA, beta = PER_BETA_START) {
      if (self$size < batch_size) {
        stop("Not enough experiences in buffer")
      }
      
      # Calculate probabilities
      priorities <- self$priorities[1:self$size]
      probabilities <- priorities^alpha / sum(priorities^alpha)
      
      # Sample indices
      indices <- sample(1:self$size, batch_size, prob = probabilities, replace = TRUE)
      
      # Calculate importance sampling weights
      weights <- (self$size * probabilities[indices])^(-beta)
      weights <- weights / max(weights)
      
      # Extract experiences
      states <- do.call(rbind, lapply(indices, function(i) self$buffer[[i]]$state))
      actions <- sapply(indices, function(i) self$buffer[[i]]$action)
      rewards <- sapply(indices, function(i) self$buffer[[i]]$reward)
      next_states <- do.call(rbind, lapply(indices, function(i) self$buffer[[i]]$next_state))
      dones <- sapply(indices, function(i) self$buffer[[i]]$done)
      
      return(list(
        states = states,
        actions = actions,
        rewards = rewards,
        next_states = next_states,
        dones = dones,
        indices = indices,
        weights = weights
      ))
    },
    
    update_priorities = function(indices, td_errors) {
      for (i in seq_along(indices)) {
        self$priorities[indices[i]] <- abs(td_errors[i]) + PER_EPSILON
      }
    }
  )
)

## TRAINING AGENT =======================================================
train_dqn_agent <- function(environment, use_per = FALSE, save_path = NULL) {
  model <- create_dqn_model(STATE_DIMENSION, 3)
  target_model <- create_dqn_model(STATE_DIMENSION, 3)
  target_model$set_weights(model$get_weights())
  
  if (use_per) {
    replay_buffer <- PrioritizedReplayBuffer$new(REPLAY_BUFFER_SIZE)
  } else {
    replay_buffer <- list()
  }
  
  episode_rewards <- numeric(TRAINING_EPISODES)
  episode_equities <- numeric(TRAINING_EPISODES)
  
  cat(sprintf("Training %s agent...\n", if (use_per) "PER DQN" else "DQN"))
  
  pb <- progress_bar$new(
    format = "  Episode :ep/:total [:bar] :percent in :elapsed | ETA: :eta",
    total = TRAINING_EPISODES, clear = FALSE, width = 60
  )
  
  for (episode in 1:TRAINING_EPISODES) {
    state <- environment$reset()
    total_reward <- 0
    step_count <- 0
    
    # Epsilon decay
    epsilon <- EPSILON_END + (EPSILON_START - EPSILON_END) * 
               exp(-episode / EPSILON_DECAY_EPISODES)
    
    max_steps_ep <- environment$max_steps - environment$current_step
    pb_steps <- progress_bar$new(
        format = "    Step :step/:total [:bar] :percent",
        total = max_steps_ep, clear = FALSE, width = 50
    )
    
    while (!environment$is_done()) {
      pb_steps$tick()
      # Action selection
      if (runif(1) < epsilon) {
        action <- sample(0:2, 1)
      } else {
        q_values <- predict(model, matrix(state, nrow = 1), verbose = 0)
        action <- which.max(q_values[1,]) - 1
      }
      
      # Environment step
      step_result <- environment$step(action)
      next_state <- step_result$next_state
      reward <- step_result$reward
      done <- step_result$done
      
      # Store experience
      if (use_per) {
        replay_buffer$add(state, action, reward, next_state, done)
      } else {
        replay_buffer[[length(replay_buffer) + 1]] <- list(
          state = state,
          action = action,
          reward = reward,
          next_state = next_state,
          done = done
        )
        if (length(replay_buffer) > REPLAY_BUFFER_SIZE) {
          replay_buffer <- replay_buffer[-1]
        }
      }
      
      # Training
      if ((use_per && replay_buffer$size >= WARMUP_STEPS) || 
          (!use_per && length(replay_buffer) >= WARMUP_STEPS)) {
        
        if (use_per) {
          beta <- PER_BETA_START + (PER_BETA_END - PER_BETA_START) * 
                  (episode / TRAINING_EPISODES)
          batch <- replay_buffer$sample(BATCH_SIZE, PER_ALPHA, beta)
        } else {
          indices <- sample(length(replay_buffer), BATCH_SIZE)
          batch <- list(
            states = do.call(rbind, lapply(indices, function(i) replay_buffer[[i]]$state)),
            actions = sapply(indices, function(i) replay_buffer[[i]]$action),
            rewards = sapply(indices, function(i) replay_buffer[[i]]$reward),
            next_states = do.call(rbind, lapply(indices, function(i) replay_buffer[[i]]$next_state)),
            dones = sapply(indices, function(i) replay_buffer[[i]]$done),
            weights = rep(1, BATCH_SIZE)
          )
        }
        
        # Double DQN training
        current_q_values <- predict(model, batch$states, verbose = 0)
        next_q_values_online <- predict(model, batch$next_states, verbose = 0)
        next_q_values_target <- predict(target_model, batch$next_states, verbose = 0)
        
        # Select actions using online network, evaluate using target network
        next_actions <- apply(next_q_values_online, 1, which.max)
        next_q_values <- next_q_values_target[cbind(1:BATCH_SIZE, next_actions)]
        
        # Calculate targets
        targets <- batch$rewards + GAMMA * next_q_values * (1 - batch$dones)
        
        # Update Q-values
        target_q_values <- current_q_values
        target_q_values[cbind(1:BATCH_SIZE, batch$actions + 1)] <- targets
        
        # Calculate TD errors for PER
        if (use_per) {
          td_errors <- targets - current_q_values[cbind(1:BATCH_SIZE, batch$actions + 1)]
          replay_buffer$update_priorities(batch$indices, td_errors)
        }
        
        # Train the model
        model %>% train_on_batch(batch$states, target_q_values, sample_weight = tf$convert_to_tensor(batch$weights))
      }
      
      state <- next_state
      total_reward <- total_reward + reward
      step_count <- step_count + 1
      
      if (done) break
    }
    
    # Update target network
    if (episode %% TARGET_UPDATE_FREQUENCY == 0) {
      target_model$set_weights(model$get_weights())
    }
    
    episode_rewards[episode] <- total_reward
    episode_equities[episode] <- environment$equity
    
    pb$tick(tokens = list(ep = episode))
    cat(sprintf(" | Reward=%.2f, Equity=%.2f, Epsilon=%.3f\n", 
                  total_reward, environment$equity, epsilon))
  }
  
  # Save model weights
  if (!is.null(save_path)) {
    save_model_weights_hdf5(model, save_path)
    cat(sprintf("Model weights saved to %s\n", save_path))
  }
  
  return(list(
    model = model,
    episode_rewards = episode_rewards,
    episode_equities = episode_equities
  ))
}

## EVALUATION AND BENCHMARKING ==========================================
evaluate_agent <- function(model, environment, name = "Agent") {
  state <- environment$reset()
  actions <- c()
  rewards <- c()
  equities <- c(environment$equity)
  soc_history <- c(environment$soc)
  
  while (!environment$is_done()) {
    q_values <- predict(model, matrix(state, nrow = 1), verbose = 0)
    action <- which.max(q_values[1,]) - 1
    
    step_result <- environment$step(action)
    
    actions <- c(actions, action)
    rewards <- c(rewards, step_result$reward)
    equities <- c(equities, environment$equity)
    soc_history <- c(soc_history, environment$soc)
    
    state <- step_result$next_state
    
    if (step_result$done) break
  }
  
  cat(sprintf("%s final equity: %.2f\n", name, environment$equity))
  
  return(list(
    final_equity = environment$equity,
    actions = actions,
    rewards = rewards,
    equities = equities,
    soc_history = soc_history
  ))
}

run_benchmark <- function(environment, strategy = "random") {
  state <- environment$reset()
  actions <- c()
  rewards <- c()
  equities <- c(environment$equity)
  
  while (!environment$is_done()) {
    if (strategy == "random") {
      action <- sample(0:2, 1)
    } else if (strategy == "heuristic") {
      current_price <- environment$prices[environment$current_step]
      if (current_price < 0) {
        action <- 1  # Charge when price is negative
      } else if (current_price > 50) {  # Discharge when price is high
        action <- 2
      } else {
        action <- 0  # Hold otherwise
      }
    }
    
    step_result <- environment$step(action)
    
    actions <- c(actions, action)
    rewards <- c(rewards, step_result$reward)
    equities <- c(equities, environment$equity)
    
    if (step_result$done) break
  }
  
  cat(sprintf("%s strategy final equity: %.2f\n", 
              tools::toTitleCase(strategy), environment$equity))
  
  return(list(
    final_equity = environment$equity,
    actions = actions,
    rewards = rewards,
    equities = equities
  ))
}

## MAIN EXECUTION ======================================================
main <- function() {
  # Load and preprocess data
  cat("Loading and preprocessing data...\n")
  data <- load_and_preprocess_data("data_task4.csv")
  
  # Create train/test split
  total_length <- length(data$original_prices)
  train_end <- floor(0.8 * total_length)
  
  train_env <- BatteryEnvironment$new(
    data$original_prices[1:train_end],
    data$returns[1:train_end],
    data$trend[1:train_end]
  )
  
  test_env <- BatteryEnvironment$new(
    data$original_prices[(train_end + 1):total_length],
    data$returns[(train_end + 1):total_length],
    data$trend[(train_end + 1):total_length]
  )
  
  # Train baseline DQN
  cat("\n=== Training Baseline DQN ===\n")
  baseline_results <- train_dqn_agent(train_env, use_per = FALSE, 
                                     save_path = "model_weights_baseline.h5")
  
  # Train PER DQN
  cat("\n=== Training PER DQN ===\n")
  per_results <- train_dqn_agent(train_env, use_per = TRUE, 
                                save_path = "model_weights_per.h5")
  
  # Evaluate on test set
  cat("\n=== Evaluation on Test Set ===\n")
  
  # Load models for evaluation
  baseline_model <- create_dqn_model(STATE_DIMENSION, 3)
  load_model_weights_hdf5(baseline_model, "model_weights_baseline.h5")
  
  per_model <- create_dqn_model(STATE_DIMENSION, 3)
  load_model_weights_hdf5(per_model, "model_weights_per.h5")
  
  # Evaluate agents
  baseline_eval <- evaluate_agent(baseline_model, test_env, "Baseline DQN")
  per_eval <- evaluate_agent(per_model, test_env, "PER DQN")
  
  # Run benchmarks
  random_eval <- run_benchmark(test_env, "random")
  heuristic_eval <- run_benchmark(test_env, "heuristic")
  
  # Create comparison plot
  max_len <- max(length(baseline_eval$equities), length(per_eval$equities),
                 length(random_eval$equities), length(heuristic_eval$equities))
  
  plot_data <- data.frame(
    Step = 1:max_len,
    Baseline_DQN = c(baseline_eval$equities, rep(NA, max_len - length(baseline_eval$equities))),
    PER_DQN = c(per_eval$equities, rep(NA, max_len - length(per_eval$equities))),
    Random = c(random_eval$equities, rep(NA, max_len - length(random_eval$equities))),
    Heuristic = c(heuristic_eval$equities, rep(NA, max_len - length(heuristic_eval$equities)))
  )
  
  p <- ggplot(plot_data, aes(x = Step)) +
    geom_line(aes(y = Baseline_DQN, color = "Baseline DQN"), size = 1) +
    geom_line(aes(y = PER_DQN, color = "PER DQN"), size = 1) +
    geom_line(aes(y = Random, color = "Random Policy"), size = 1) +
    geom_line(aes(y = Heuristic, color = "Heuristic Policy"), size = 1) +
    labs(title = "Battery Trading Agent Performance Comparison",
         x = "Time Steps", y = "Equity (EUR)",
         color = "Strategy") +
    theme_minimal() +
    scale_color_manual(values = c(
      "Baseline DQN" = "blue",
      "PER DQN" = "red",
      "Random Policy" = "gray",
      "Heuristic Policy" = "green"
    ))
  
  print(p)
  
  # Print final results
  cat("\n=== Final Results ===\n")
  cat(sprintf("Baseline DQN: %.2f EUR\n", baseline_eval$final_equity))
  cat(sprintf("PER DQN: %.2f EUR\n", per_eval$final_equity))
  cat(sprintf("Random Policy: %.2f EUR\n", random_eval$final_equity))
  cat(sprintf("Heuristic Policy: %.2f EUR\n", heuristic_eval$final_equity))
  
  return(list(
    baseline_results = baseline_results,
    per_results = per_results,
    evaluations = list(
      baseline = baseline_eval,
      per = per_eval,
      random = random_eval,
      heuristic = heuristic_eval
    )
  ))
}

# Note: You need to install R6 package for the object-oriented approach
if (!requireNamespace("R6", quietly = TRUE)) install.packages("R6")
library(R6)

# Run the main function
results <- main()
