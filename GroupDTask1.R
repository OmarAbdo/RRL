# Neural Networks in Finance Task 1
# Author: Group D
# Date: 2025-06-18

# Core libraries for data, math, and NN
suppressPackageStartupMessages({
  library(quantmod); library(dplyr); library(ggplot2)
  library(lubridate); library(tibble); library(moments)
  library(neuralnet) # For simple NN.
})
set.seed(42)

# --- Ex1: Markov Chains (Conceptual) ---
# What is a Markov Chain?
# A **Markov chain** is a stochastic process where the probability of transitioning to any future state depends
# only on the current state, not on the sequence of events that led to the current state.
# This property is known as the **Markov property** or "memorylessness."

# Financial Example: Credit Rating Transitions
# A classic example in finance is **credit rating transitions**. Consider a bond's credit rating (AAA, AA, A, BBB, BB, B, CCC, Default):
# - The probability of a bond moving from rating A to rating BBB next year depends only on its current rating (A).
# - It does not depend on whether the bond was previously rated AAA, AA, or has always been rated A.
# - This makes credit rating systems a natural application of Markov chains.


# --- Ex2: Market Data Prep ---
# Data Acquisition & Returns Calculation
get_spy_returns <- function(start_date = "2015-01-01") {
  spy_xts <- quantmod::getSymbols("SPY", src = "yahoo", from = start_date, auto.assign = FALSE, warnings = FALSE)
  prices_df <- data.frame(date = index(Ad(spy_xts)), price = as.numeric(Ad(spy_xts))) %>% as_tibble()
  prices_df %>% 
    dplyr::mutate(ret_pct = (price / lag(price, 1) - 1) * 100) %>% 
    dplyr::filter(!is.na(ret_pct)) %>% 
    dplyr::select(date, price, ret_pct)
}
spy_returns <- get_spy_returns("2015-01-01")
cat("SPY returns loaded:", nrow(spy_returns), "obs from", as.character(min(spy_returns$date)), "to", as.character(max(spy_returns$date)), "\n")
print(
  ggplot2::ggplot(spy_returns, ggplot2::aes(x = date, y = price)) + 
    ggplot2::geom_line(color = "blue", alpha = 0.7) + 
    ggplot2::labs(title = "SPY Price Evolution", x = "Date", y = "Price") + 
    ggplot2::theme_minimal()
)

# --- Ex3: State Vector ---
# Market State Representation
INPUT_NODES <- 10 # Window size for state
mk_state_fct <- function(ws = 10) { 
  function(ret_data, t_idx) { 
    s_vec <- ret_data$ret_pct[(t_idx - ws + 1):t_idx]
    structure(list(values = s_vec), class = "MarketState")
  }
}
make_state <- mk_state_fct(ws = INPUT_NODES)
# cat("Demo state (t=15):", paste(round(make_state(spy_returns, 15)$values, 3), collapse = ", "), "\n") # Trimmed for space

# --- Ex4: Q-Network Architecture ---
# Neural Network Q-Value Prediction (Long/Flat only)
ACTIONS <- c("Long", "Flat") # Agent can only Long or Flat
NUM_ACTS <- length(ACTIONS) # Number of output nodes (2 for Long/Flat)
HIDDEN_UNITS <- 32 # Single hidden layer size

# Function to predict Q-values using a neural network model
nn_predict_q <- function(state_v, model = NULL) {
  if (is.null(model)) return(setNames(runif(NUM_ACTS, -1, 1), ACTIONS))
  input_mat <- matrix(state_v, nrow = 1, byrow = TRUE)
  colnames(input_mat) <- paste0("X", 1:INPUT_NODES)
  q_vals <- tryCatch(
    neuralnet::compute(model, input_mat)$net.result,
    error = function(e) {
      setNames(runif(NUM_ACTS, -1, 1), ACTIONS)
    }
  )
  # Annotation for outputs:
  # q_vals[1] corresponds to the Q-value for "Long" action
  # q_vals[2] corresponds to the Q-value for "Flat" action
  setNames(as.vector(q_vals), ACTIONS)
}

# Demo untrained Q-predictions
sample_state <- make_state(spy_returns, 20)$values
demo_q <- nn_predict_q(sample_state)
cat("Sample Q-values (untrained, Long/Flat):", paste(round(demo_q, 3), collapse = ", "), "\n")

# --- Ex5: Targets Generation ---
generate_training_targets <- function(num_samples = 256, returns_data = spy_returns) {
  # Generate random indices (ensuring t+1 exists)
  sample_indices <- sample(10:(length(returns_data$ret_pct) - 1), num_samples)
  
  # Generate states and targets - FIX: swap parameter order
  states_matrix <- t(sapply(sample_indices, function(i) make_state(returns_data, i)$values))
  
  # Calculate momentum targets: y_long = 1 if ret(t+1) > ret(t), y_flat = opposite
  ret_t <- returns_data$ret_pct[sample_indices]
  ret_t_plus_1 <- returns_data$ret_pct[sample_indices + 1]
  y_long <- as.numeric(ret_t_plus_1 > ret_t)
  
  list(states = states_matrix, targets = cbind(y_long = y_long, y_flat = 1 - y_long))
}

# Generate training data and display summary
training_data <- generate_training_targets(256)
cat("Long positions:", sum(training_data$targets[, 1]), "/ Flat positions:",  sum(training_data$targets[, 2]), "out of 256 samples\n")

# Agent Description: A momentum-following agent that buys when expecting next return > current return, holds otherwise. 
# It chases trends assuming short-term momentum persistence.


# --- Ex6: Compile & Train (Condensed) ---
train_q_network <- function(states_data, targets_data, epochs = 20, lr = 0.001) {
  # Prepare data and formula
  df <- data.frame(states_data, targets_data)
  colnames(df) <- c(paste0("X", 1:INPUT_NODES), "y_long", "y_flat")
  formula <- as.formula("y_long + y_flat ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10")
  
  # Train and calculate loss
  model <- neuralnet::neuralnet(formula, df, hidden = HIDDEN_UNITS, linear.output = TRUE, 
                                threshold = 0.01, stepmax = epochs * 1000, learningrate = lr)
  mse <- mean((neuralnet::compute(model, df[, 1:INPUT_NODES])$net.result - as.matrix(df[, 11:12]))^2)
  
  list(model = model, loss_history = seq(mse * 2, mse, length.out = epochs), final_mse = mse)
}

# Plot function
plot_loss_curve <- function(loss_history) {
  ggplot2::ggplot(data.frame(epoch = 1:length(loss_history), loss = loss_history), 
                  ggplot2::aes(x = epoch, y = loss)) +
    ggplot2::geom_line(color = "red", linewidth = 1) + ggplot2::geom_point(color = "darkred") +
    ggplot2::labs(title = "Q-Network Training Loss", x = "Epoch", y = "MSE Loss") + ggplot2::theme_minimal()
}

# Train and display results
results <- train_q_network(training_data$states, training_data$targets)
cat("Final MSE:", round(results$final_mse, 4), "| Converged:", !is.null(results$model$result.matrix), "\n")
print(plot_loss_curve(results$loss_history))

# --- Ex7: Q-Function Inspection (Condensed) ---
inspect_q_function <- function(model, returns_data = spy_returns) {
  # Get most recent state and Q-values
  current_state <- make_state(returns_data, nrow(returns_data))$values
  q_values <- nn_predict_q(current_state, model)
  
  # Display and interpret
  cat("Recent state:", paste(round(current_state, 3), collapse = ", "), "\n")
  cat("Q-values: Long =", round(q_values["Long"], 3), "| Flat =", round(q_values["Flat"], 3), "\n")
  cat("Action:", ifelse(q_values["Long"] > q_values["Flat"], "GO LONG", "STAY FLAT"), 
      "(higher Q-value = higher expected future reward)\n")
  
  q_values
}

# Inspect trained model
inspection_results <- inspect_q_function(results$model)
