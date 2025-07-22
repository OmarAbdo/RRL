# Minimal script to reproduce the TensorFlow placeholder error
library(keras)
library(tensorflow)

# Set seed for reproducibility
set.seed(42)
tf$random$set_seed(42)

# 1. Define a simple model
state_size <- 10
action_size <- 3
hidden_units <- 16

model <- keras_model_sequential() %>%
  layer_dense(units = hidden_units, activation = "relu", input_shape = c(state_size)) %>%
  layer_dense(units = action_size, activation = "linear")

model %>% compile(
  optimizer = tf$keras$optimizers$legacy$Adam(learning_rate = 0.001),
  loss = "mse"
)

# 2. Generate dummy data that mimics the structure of the real data
batch_size <- 64
states <- array(rnorm(batch_size * state_size), dim = c(batch_size, state_size))
actions <- sample(0:(action_size - 1), batch_size, replace = TRUE)
rewards <- rnorm(batch_size)
next_states <- array(rnorm(batch_size * state_size), dim = c(batch_size, state_size))
dones <- sample(c(TRUE, FALSE), batch_size, replace = TRUE)

# 3. Replicate the target calculation logic
next_q_values <- apply(predict(model, next_states, verbose = 0), 1, max)
targets <- rewards + 0.99 * next_q_values * (!dones)

current_q_py <- predict(model, states, verbose = 0)
current_q <- as.matrix(current_q_py)
for (i in 1:length(actions)) {
  current_q[i, actions[i] + 1] <- targets[i]
}

# 4. Attempt the train_on_batch call
cat("Attempting train_on_batch...\n")
tryCatch({
  # Explicitly cast to float32 to prevent potential dtype issues
  states_tensor <- tf$cast(states, dtype = "float32")
  q_tensor <- tf$cast(current_q, dtype = "float32")
  
  train_on_batch(model, states_tensor, q_tensor)
  cat("train_on_batch succeeded.\n")
}, error = function(e) {
  cat("train_on_batch failed with error:\n")
  print(e)
})
