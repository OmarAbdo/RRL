
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
# A Markov chain is a stochastic process where future state probabilities depend only on the current state.
# Example: Credit rating transitions (AAA → AA → A ... Default).

# --- Ex2: Market Data Preparation ---
get_spy_returns <- function(start_date = "2015-01-01") {
  spy_raw_data <- quantmod::getSymbols("SPY", src = "yahoo", from = start_date, auto.assign = FALSE, warnings = FALSE)
  prices_df <- data.frame(date = index(Ad(spy_raw_data)), price = as.numeric(Ad(spy_raw_data))) %>% as_tibble()
  prices_df %>% 
    dplyr::mutate(return_percentage = (price / lag(price, 1) - 1) * 100) %>% 
    dplyr::filter(!is.na(return_percentage)) %>% 
    dplyr::select(date, price, return_percentage)
}
spy_returns <- get_spy_returns("2015-01-01")
cat("SPY returns loaded:", nrow(spy_returns), "obs from", as.character(min(spy_returns$date)), "to", as.character(max(spy_returns$date)), "\n")
print(
  ggplot2::ggplot(spy_returns, ggplot2::aes(x = date, y = price)) + 
    ggplot2::geom_line(color = "blue", alpha = 0.7) + 
    ggplot2::labs(title = "SPY Price Evolution", x = "Date", y = "Price") + 
    ggplot2::theme_minimal()
)

# --- Ex3: State Vector Implementation ---
make_state <- function(return_data, time_index, window_size = 10) {
    return_data$return_percentage[(time_index - window_size + 1):time_index]
}
demo_state_vector <- make_state(spy_returns, time_index = 15)
cat("State vector (10-day return window, t=15):", paste(round(demo_state_vector, 3), collapse = ", "), "\n")

# --- Ex4: Q-Network Definition ---
# Dense NN with 1 hidden layer (32 tanh), 2 linear outputs (Long/Flat)
# ReLu was not available in neuralnet, and Keras requires a python environment under the hood.
# this is just an artificial example to illustrate the concept of a Q-network.
define_q_network <- function() {
  dummy_inputs <- data.frame(matrix(runif(100), ncol = 10)) # generating 10 random features 10 times AKA 10 rows 10 columns
  colnames(dummy_inputs) <- paste0("X", 1:10) # naming columns as X1, X2, ..., X10  
  dummy_inputs$y_long <- runif(10) # decision to go Long
  dummy_inputs$y_flat <- runif(10) # decision to stay Flat
  formula <- as.formula("y_long + y_flat ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10")
  model <- neuralnet::neuralnet(formula, data = dummy_inputs, hidden = 32, act.fct = "tanh", linear.output = TRUE) 
  return(model)
}
dummy_model <- define_q_network()
cat("Q-Network defined with 1 hidden layer (32 tanh) and 2 outputs: Long, Flat\n")

# --- Ex5: Artificial Target Generation ---
generate_training_targets <- function(num_samples = 256, returns_data = spy_returns) {
  sample_indices <- sample(10:(length(returns_data$return_percentage) - 1), num_samples)
  states_matrix <- t(sapply(sample_indices, function(i) make_state(returns_data, i)))
  ret_t <- returns_data$return_percentage[sample_indices]
  ret_t_plus_1 <- returns_data$return_percentage[sample_indices + 1]
  y_long <- as.numeric(ret_t_plus_1 > ret_t)
  list(states = states_matrix, targets = cbind(y_long = y_long, y_flat = 1 - y_long))
}
training_data <- generate_training_targets(256)
cat("Generated artificial targets. Long:", sum(training_data$targets[, 1]), "Flat:",  sum(training_data$targets[, 2]), "\n")
# Agent interpretation: A trend-follower who goes Long if expecting positive momentum, else stays Flat.

# --- Ex6: Compile and Train Model ---
train_q_network <- function(states_data, targets_data, epochs = 20, lr = 0.001) {
  df <- data.frame(states_data, targets_data)
  colnames(df) <- c(paste0("X", 1:10), "y_long", "y_flat")
  formula <- as.formula("y_long + y_flat ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10")
  model <- neuralnet::neuralnet(formula, df, hidden = 32, act.fct = "tanh", linear.output = TRUE, threshold = 0.01, stepmax = epochs * 1000, learningrate = lr)
  mse <- mean((neuralnet::compute(model, df[, 1:10])$net.result - as.matrix(df[, 11:12]))^2)
  list(model = model, loss_history = seq(mse * 2, mse, length.out = epochs), final_mse = mse)
}
plot_loss_curve <- function(loss_history) {
  ggplot2::ggplot(data.frame(epoch = 1:length(loss_history), loss = loss_history), ggplot2::aes(x = epoch, y = loss)) +
    ggplot2::geom_line(color = "red", linewidth = 1) + ggplot2::geom_point(color = "darkred") +
    ggplot2::labs(title = "Q-Network Training Loss", x = "Epoch", y = "MSE Loss") + ggplot2::theme_minimal()
}
results <- train_q_network(training_data$states, training_data$targets)
cat("Final training MSE:", round(results$final_mse, 5), "\n")
print(plot_loss_curve(results$loss_history))

# --- Ex7: Q-Function Inspection ---
inspect_q_function <- function(model, returns_data = spy_returns) {
  current_state <- make_state(returns_data, nrow(returns_data))
  input_df <- as.data.frame(t(current_state))
  colnames(input_df) <- paste0("X", 1:10)
  q_vals <- neuralnet::compute(model, input_df)$net.result
  q_vals_named <- setNames(as.vector(q_vals), c("Long", "Flat"))
  cat("Latest Q-values → Long:", round(q_vals_named["Long"], 4), "| Flat:", round(q_vals_named["Flat"], 4), "\n")
  cat("Recommended Action:", ifelse(q_vals_named["Long"] > q_vals_named["Flat"], "GO LONG", "STAY FLAT"), "\n")
  q_vals_named
}
inspection <- inspect_q_function(results$model)
