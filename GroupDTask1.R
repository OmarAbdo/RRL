# Neural Networks in Finance Task 1
# Author: Group D
# Date: 2025-06-18

# Core libraries
suppressPackageStartupMessages({
  library(quantmod)
  library(dplyr)
  library(ggplot2)
  library(lubridate)
  library(tibble)
  library(moments)
  library(keras) # Replaced neuralnet with keras
})
set.seed(42)
# Ensure Keras (with TensorFlow backend) is installed and configured.
# If you encounter "Valid installation of TensorFlow not found" or "No module named 'tensorflow'",
# run the following commands in your R console:
# library(keras)
# install_keras(method = "conda", python_version = "3.10")
# After installation, you might need to restart your R session.

# --- Ex1: Markov Chains ---
# Stochastic process: future state depends only on current state.
# Finance example: Credit rating transitions. Going from one rating to another (e.g., AAA to AA) is independent of past ratings.

# --- Ex2: Market Data Preparation ---
get_spy_returns <- function(start_date = "2015-01-01") {
  raw_data <- quantmod::getSymbols("SPY", src = "yahoo", from = start_date, auto.assign = FALSE, warnings = FALSE)
  prices_df <- data.frame(date = index(Ad(raw_data)), price = as.numeric(Ad(raw_data))) %>% as_tibble()
  prices_df %>%
    dplyr::mutate(return_percentage = (price / lag(price, 1) - 1) * 100) %>%
    dplyr::filter(!is.na(return_percentage)) %>%
    dplyr::select(date, price, return_percentage)
}
spy_returns <- get_spy_returns("2015-01-01")
cat("SPY returns loaded:", nrow(spy_returns), "obs.\n")
print(ggplot2::ggplot(spy_returns, ggplot2::aes(x = date, y = price)) +
  ggplot2::geom_line(color = "blue", alpha = 0.7) +
  ggplot2::labs(title = "SPY Price", x = "Date", y = "Price") +
  ggplot2::theme_minimal())

# --- Ex3: State Vector Implementation ---
make_state <- function(return_data, time_index, window_size = 10) {
  return_data$return_percentage[(time_index - window_size + 1):time_index]
}
demo_state_vector <- make_state(spy_returns, time_index = 15)
cat("State (t=15):", paste(round(demo_state_vector, 3), collapse = ", "), "\n")

# --- Ex4: Q-Network Architecture Design (Keras) ---
# Design: dense net with 1 hidden layer (32 units, ReLU), 2 linear outputs.
ACTIONS <- c("Long", "Flat")
INPUT_NODES <- 10
OUTPUT_NODES <- length(ACTIONS)
HIDDEN_UNITS <- 32

# Define the Keras sequential model
q_network_model <- keras_model_sequential() %>%
  layer_dense(units = HIDDEN_UNITS, activation = "relu", input_shape = c(INPUT_NODES)) %>% # Corrected: input_input_shape to input_shape
  layer_dense(units = OUTPUT_NODES, activation = "linear")

# Annotation of outputs:
# - First output: Q-value for "Long" action.
# - Second output: Q-value for "Flat" action.
cat("Keras Q-Network model defined in 'q_network_model'.\n")

# --- Ex5: Artificial Target Generation ---
generate_training_targets <- function(num_samples = 256, returns_data = spy_returns) {
  sample_idx <- sample(10:(length(returns_data$return_percentage) - 1), num_samples)
  states_mat <- t(sapply(sample_idx, function(i) make_state(returns_data, i)))
  return_t <- returns_data$return_percentage[sample_idx]
  return_t_plus_1 <- returns_data$return_percentage[sample_idx + 1]
  y_long <- as.numeric(return_t_plus_1 > return_t)
  list(states = states_mat, targets = cbind(y_long = y_long, y_flat = 1 - y_long))
}
training_data <- generate_training_targets(256)
cat("Generated targets. Long:", sum(training_data$targets[, 1]), "Flat:", sum(training_data$targets[, 2]), "\n")
# Agent interpretation: A trend-follower.

# --- Ex6: Compile and Train Model ---
train_q_network <- function(states_data, targets_data, epochs = 20, lr = 0.001, model_to_train = q_network_model) {
  optimizer <- optimizer_adam(learning_rate = lr)
  model_to_train %>% compile(optimizer = optimizer, loss = "mse")
  history <- model_to_train %>% fit(
    x = states_data,
    y = targets_data,
    epochs = epochs,
    verbose = 0 # Suppress verbose output during training
  )
  list(model = model_to_train, loss_history = history$metrics$loss, final_mse = tail(history$metrics$loss, 1))
}
plot_loss_curve <- function(loss_history) {
  ggplot2::ggplot(data.frame(epoch = 1:length(loss_history), loss = loss_history), ggplot2::aes(x = epoch, y = loss)) +
    ggplot2::geom_line(color = "red", linewidth = 1) +
    ggplot2::geom_point(color = "darkred") +
    ggplot2::labs(title = "Q-Net Loss", x = "Epoch", y = "MSE") +
    ggplot2::theme_minimal()
}
results <- train_q_network(training_data$states, training_data$targets)
cat("Final training MSE:", round(results$final_mse, 5), "\n")
print(plot_loss_curve(results$loss_history))

# --- Ex7: Q-Function Inspection ---
nn_predict_q <- function(state_v, model) {
  input_for_predict <- array_reshape(state_v, c(1, INPUT_NODES)) # Reshape for Keras
  q_vals <- model %>% predict(input_for_predict, verbose = 0)
  setNames(as.vector(q_vals), ACTIONS)
}
inspect_q_function <- function(model, returns_data = spy_returns) {
  current_state <- make_state(returns_data, nrow(returns_data))
  q_vals_named <- nn_predict_q(current_state, model)
  cat("Q-values → Long:", round(q_vals_named["Long"], 4), "| Flat:", round(q_vals_named["Flat"], 4), "\n")
  cat("Action:", ifelse(q_vals_named["Long"] > q_vals_named["Flat"], "GO LONG", "STAY FLAT"), "\n")
  # Q-value: expected future reward for action from state.
  q_vals_named
}
inspection <- inspect_q_function(results$model)
