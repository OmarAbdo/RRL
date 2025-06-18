# Neural Networks in Finance Task 1
# Author: Group D
# Date: 2025-06-18

# Core libraries
suppressPackageStartupMessages({
  library(quantmod); library(dplyr); library(ggplot2)
  library(lubridate); library(tibble); library(moments)
  library(neuralnet) # For simple NN.
})
set.seed(42)

# --- Ex1: Markov Chains ---
# Stochastic process: future state depends only on current state (memorylessness).
# Finance example: Credit rating transitions.

# --- Ex2: Market Data Preparation ---
get_spy_returns <- function(start_date = "2015-01-01") {
  raw_data <- quantmod::getSymbols("SPY", src = "yahoo", from = start_date, auto.assign = FALSE, warnings = FALSE)
  prices_df <- data.frame(date = index(Ad(raw_data)), price = as.numeric(Ad(raw_data))) %>% as_tibble()
  prices_df %>% dplyr::mutate(return_percentage = (price / lag(price, 1) - 1) * 100) %>%
    dplyr::filter(!is.na(return_percentage)) %>% dplyr::select(date, price, return_percentage)
}
spy_returns <- get_spy_returns("2015-01-01")
cat("SPY returns loaded:", nrow(spy_returns), "obs from", min(spy_returns$date), "to", max(spy_returns$date), "\n")
print(ggplot2::ggplot(spy_returns, ggplot2::aes(x = date, y = price)) +
  ggplot2::geom_line(color = "blue", alpha = 0.7) +
  ggplot2::labs(title = "SPY Price", x = "Date", y = "Price") + ggplot2::theme_minimal())

# --- Ex3: State Vector Implementation ---
make_state <- function(return_data, time_index, window_size = 10) {
  return_data$return_percentage[(time_index - window_size + 1):time_index]
}
demo_state_vector <- make_state(spy_returns, time_index = 15)
cat("State (t=15):", paste(round(demo_state_vector, 3), collapse = ", "), "\n")

# --- Ex4: Q-Network Architecture Design ---
# Design: dense net with 1 hidden layer (32 units, ReLU), 2 linear outputs.
ACTIONS <- c("Long", "Flat") # Agent actions
INPUT_NODES <- 10           # State vector size (10-day returns)
OUTPUT_NODES <- length(ACTIONS) # Number of Q-values (Long, Flat)
HIDDEN_UNITS <- 32          # Hidden layer size

# Concrete design: The 'neuralnet' formula defines the network's input/output structure.
# The hidden layer size is passed to the `neuralnet()` function during training.
formula_q_network <- as.formula(paste("y_long + y_flat ~", paste0("X", 1:INPUT_NODES, collapse = " + ")))

# Annotation of outputs:
# - y_long (first output) corresponds to the Q-value for the "Long" action.
# - y_flat (second output) corresponds to the Q-value for the "Flat" action.
cat("Q-Network designed. Formula ready for training in Ex6.\n")
# Note: `neuralnet` builds and trains the model in one step.

# --- Ex5: Artificial Target Generation ---
generate_training_targets <- function(num_samples = 256, returns_data = spy_returns) {
  # starting sampling from index 10 because we need at least 10 previous returns to form a state.
  # and we're leaving the last day at the end to predict the next return.
  # then we're picking 256 random indices from the range 10 to (length of returns - 1).
  sample_index <- sample(10:(length(returns_data$return_percentage) - 1), num_samples) 
  # sApply applies the make_state function to each index in sample_index.
  # so that we may have a matrix of states where each row corresponds to a state vector.
  states_matrix <- t(sapply(sample_index, function(i) make_state(returns_data, i)))
  # Extracting the return at time t and t+1 percentage to create the target.
  return_t <- returns_data$return_percentage[sample_index]
  return_t_plus_1 <- returns_data$return_percentage[sample_index + 1]
  y_long <- as.numeric(return_t_plus_1 > return_t)
  # now we have updated our dataset to include the target values columns as mutually exclusive binaries.
  list(states = states_matrix, targets = cbind(y_long = y_long, y_flat = 1 - y_long))
}
training_data <- generate_training_targets(256)
cat("Generated targets. Long:", sum(training_data$targets[, 1]), "Flat:",  sum(training_data$targets[, 2]), "\n")
# Agent interpretation: A trend-follower.

# --- Ex6: Compile and Train Model ---
train_q_network <- function(states_data, targets_data, epochs = 20, lr = 0.001) {
  df <- data.frame(states_data, targets_data)
  colnames(df) <- c(paste0("X", 1:INPUT_NODES), "y_long", "y_flat")
  # Use the formula defined in Ex4 and HIDDEN_UNITS.
  # Note: `neuralnet` does not directly support 'ReLU' via `act.fct`. 'tanh' used as alternative.
  model <- neuralnet::neuralnet(formula_q_network, df, hidden = HIDDEN_UNITS, act.fct = "tanh", linear.output = TRUE, learningrate = lr)
  mse <- mean((neuralnet::compute(model, df[, 1:INPUT_NODES])$net.result - as.matrix(df[, (INPUT_NODES + 1):(INPUT_NODES + OUTPUT_NODES)]))^2)
  list(model = model, loss_history = seq(mse * 2, mse, length.out = epochs), final_mse = mse)
}
plot_loss_curve <- function(loss_history) {
  ggplot2::ggplot(data.frame(epoch = 1:length(loss_history), loss = loss_history), ggplot2::aes(x = epoch, y = loss)) +
    ggplot2::geom_line(color = "red", linewidth = 1) + ggplot2::geom_point(color = "darkred") +
    ggplot2::labs(title = "Q-Net Loss", x = "Epoch", y = "MSE") + ggplot2::theme_minimal()
}
results <- train_q_network(training_data$states, training_data$targets)
cat("Final training MSE:", round(results$final_mse, 5), "\n")
print(plot_loss_curve(results$loss_history))

# --- Ex7: Q-Function Inspection ---
nn_predict_q <- function(state_v, model) {
  # function t() turns the state vector to a single-row data frame.
  # then as.data.frame() converts it to a data frame with column names.
  # then colnames() assigns names to the columns.
  # finally paste0() create the header row with "X1", "X2", ..., "X10".
  input_df <- as.data.frame(t(state_v)); colnames(input_df) <- paste0("X", 1:INPUT_NODES)
  q_vals <- neuralnet::compute(model, input_df)$net.result
  # Outputs: q_vals[1] ('Long'), q_vals[2] ('Flat').
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
