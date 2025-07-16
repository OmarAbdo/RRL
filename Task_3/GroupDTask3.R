
# Group D - Task 3

# --- Exercise 1: Transfer from previous tasks ---
suppressPackageStartupMessages({
  library(quantmod)
  library(dplyr)
  library(ggplot2)
  library(keras)
  library(tibble)
})

set.seed(42)

# Load Data
getSymbols("SPY", src = "yahoo", from = "2015-01-01", auto.assign = TRUE)
spy_returns_xts <- dailyReturn(Ad(SPY)) * 100
returns <- as.numeric(spy_returns_xts)

# --- Exercise 2: Data Split ---
train_size <- floor(0.8 * length(returns))
train_returns <- returns[1:train_size]
test_returns <- returns[(train_size + 1):length(returns)]

# Environment Functions
make_state <- function(t, pos, returns_data, window_size = 10) {
  state_vector <- returns_data[(t - window_size + 1):t]
  c(state_vector, pos)
}

env_reset <- function(returns_data) {
  list(t = 10, pos = 0, equity = 1, done = FALSE, returns_data = returns_data)
}

env_step <- function(env, action, transaction_cost = 0.0005) {
  t <- env$t
  pos <- env$pos
  equity <- env$equity
  returns_data <- env$returns_data

  if (t >= length(returns_data) - 1) {
    return(list(next_env = env, reward = 0, obs = rep(0, 11), done = TRUE))
  }

  r_t1 <- returns_data[t + 1] / 100
  reward <- 0

  if (action == 1 && pos == 0) { # Buy
    reward <- r_t1 - transaction_cost
    pos <- 1
  } else if (action == 0 && pos == 1) { # Sell
    reward <- -transaction_cost
    pos <- 0
  } else if (action == 1 && pos == 1) { # Hold
    reward <- r_t1
  } # if action is 0 and pos is 0, reward is 0

  equity <- equity * (1 + reward)
  t <- t + 1
  done <- (t >= length(returns_data) - 1)
  
  next_env <- list(t = t, pos = pos, equity = equity, done = done, returns_data = returns_data)
  obs <- make_state(t, pos, returns_data)
  
  list(next_env = next_env, reward = reward, obs = obs, done = done)
}

# Replay Buffer
replay_buffer <- new.env()
buffer_capacity <- 5000
state_size <- 11

replay_buffer$s  <- matrix(0, nrow = buffer_capacity, ncol = state_size)
replay_buffer$a  <- integer(buffer_capacity)
replay_buffer$r  <- numeric(buffer_capacity)
replay_buffer$s2 <- matrix(0, nrow = buffer_capacity, ncol = state_size)
replay_buffer$done <- logical(buffer_capacity)
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
  if (buffer$idx > buffer_capacity) {
    buffer$idx <- 1
    buffer$is_full <- TRUE
  }
}

sample_batch <- function(buffer, n) {
  max_index <- if (buffer$is_full) buffer_capacity else (buffer$idx - 1)
  sampled_idx <- sample(1:max_index, size = n, replace = FALSE)
  
  list(
    s = buffer$s[sampled_idx, , drop = FALSE],
    a = buffer$a[sampled_idx],
    r = buffer$r[sampled_idx],
    s2 = buffer$s2[sampled_idx, , drop = FALSE],
    done = buffer$done[sampled_idx]
  )
}

# --- Exercise 3: Network Factory ---
build_qnet <- function(input_nodes = 11, output_nodes = 2, hidden_units = 32) {
  model <- keras_model_sequential() %>%
    layer_dense(units = hidden_units, activation = "relu", input_shape = c(input_nodes)) %>%
    layer_dense(units = output_nodes, activation = "linear")
  return(model)
}

# --- Exercise 4: Epsilon-Greedy Function ---
epsilon_decay <- function(ep, start = 1.0, final = 0.01, decay_rate = 0.01) {
  final + (start - final) * exp(-decay_rate * ep)
}

# --- Exercise 5: Hyperparameters ---
gamma <- 0.99 # Discount factor for future rewards. A high value signifies the agent values future rewards almost as much as immediate ones.
learning_rate <- 0.001 # Step size for gradient descent. A small value ensures stable learning.
replay_capacity <- 5000 # Max size of the replay buffer. Large enough to store diverse experiences.
warmup_size <- 1000 # Steps before training starts. Ensures the buffer has enough random samples.
max_steps_ep <- 200 # Max steps per episode. Prevents episodes from running too long.
epsilon_start <- 1.0 # Initial epsilon. Starts with full exploration.
epsilon_final <- 0.01 # Final epsilon. Ends with mostly exploitation.
epsilon_decay_rate <- 0.01 # Rate of epsilon decay. Controls the speed of transition from exploration to exploitation.
batch_size <- 32 # Number of samples per training batch. A standard size for stable updates.

# --- Exercise 6: Training Loop ---
online_net <- build_qnet()
target_net <- build_qnet()
target_net %>% set_weights(online_net %>% get_weights())

online_net %>% compile(
  optimizer = optimizer_adam(learning_rate = learning_rate),
  loss = "mse"
)

total_episodes <- 10
episode_log <- list()

for (ep in 1:total_episodes) {
  env <- env_reset(train_returns)
  state <- make_state(env$t, env$pos, env$returns_data)
  
  episode_reward <- 0
  episode_equity <- c(env$equity)
  
  for (step in 1:max_steps_ep) {
    # Epsilon-greedy action selection
    epsilon <- epsilon_decay(ep, start = epsilon_start, final = epsilon_final, decay_rate = epsilon_decay_rate)
    if (runif(1) < epsilon) {
      action <- sample(0:1, 1) # Explore
    } else {
      q_values <- online_net %>% predict(t(state), verbose = 0)
      action <- which.max(q_values) - 1 # Exploit (0 or 1)
    }
    
    # Environment step
    step_result <- env_step(env, action)
    next_state <- step_result$obs
    reward <- step_result$reward
    done <- step_result$done
    
    store_transition(replay_buffer, state, action, reward, next_state, done)
    
    state <- next_state
    env <- step_result$next_env
    episode_reward <- episode_reward + reward
    episode_equity <- c(episode_equity, env$equity)
    
    # Training
    if (replay_buffer$idx > warmup_size) {
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
    
    if (done) break
  }
  
  # Update target network
  if (ep %% 10 == 0) {
    target_net %>% set_weights(online_net %>% get_weights())
  }
  
  episode_log[[ep]] <- list(reward = episode_reward, equity_hist = episode_equity)
  cat(sprintf("Episode: %d, Total Reward: %.4f, Final Equity: %.4f\n", ep, episode_reward, tail(episode_equity, 1)))
}

# --- Exercise 7: Visual Inspection ---
# Run an episode with epsilon = 0 on training data
env_eval <- env_reset(train_returns)
state_eval <- make_state(env_eval$t, env_eval$pos, env_eval$returns_data)
agent_equity <- c(env_eval$equity)

while(!env_eval$done) {
  q_values <- online_net %>% predict(t(state_eval), verbose = 0)
  action <- which.max(q_values) - 1
  
  step_result <- env_step(env_eval, action)
  state_eval <- step_result$obs
  env_eval <- step_result$next_env
  agent_equity <- c(agent_equity, env_eval$equity)
}

# Buy and Hold Strategy
buy_and_hold_equity <- cumprod(1 + c(0, train_returns[11:length(train_returns)] / 100))

# Plotting
plot_df <- data.frame(
  Step = 1:length(agent_equity),
  Agent = agent_equity,
  BuyAndHold = buy_and_hold_equity[1:length(agent_equity)]
)

ggplot(plot_df, aes(x = Step)) +
  geom_line(aes(y = Agent, colour = "Agent")) +
  geom_line(aes(y = BuyAndHold, colour = "Buy and Hold")) +
  labs(title = "Agent vs. Buy and Hold Strategy", y = "Equity", x = "Time Steps") +
  scale_colour_manual("", values = c("Agent"="blue", "Buy and Hold"="red")) +
  theme_minimal()

# --- Exercise 8: Discussion ---
# Trade costs directly reduce the reward for buying or selling, making the agent
# more hesitant to trade. This can lead to a strategy that holds positions for
# longer to avoid incurring frequent costs. To improve the result, one could:
#   * Optimize transaction costs: Find a broker with lower fees.
#   * Improve the reward function: Add a term that penalizes frequent trading.
#   * Use a more sophisticated model: A more complex model might be able to
#     better identify profitable trades that justify the transaction costs.

# --- Reflection Questions ---

# Explain "bootstrapping" in the Bellman update.
# Bootstrapping, in the context of the Bellman equation, refers to the process of
# updating a value estimate based on other value estimates. In our Q-learning
# implementation, the target Q-value for a state-action pair is calculated using
# the estimated Q-values of the *next* state. We are using an existing estimate
# (the Q-value of the next state) to improve the current estimate, which is the
# essence of bootstrapping.

# Why is a separate target network more stable than using the online weights directly as target?
# A separate target network provides a stable target for the online network to learn from.
# If we were to use the online network to calculate the target values, the target
# would change with every weight update. This creates a moving-target problem,
# making the training process unstable. By using a fixed (or slowly updating)
# target network, we provide a consistent objective for the online network, which
# leads to more stable and reliable learning.

# How would you extend the action space to short selling?
# To extend the action space to include short selling, we would need to add a
# third action, for example, `action = 2` for "short". The environment's
# `env_step` function would need to be modified to handle this new action. This
# would involve:
#   * When the agent chooses to short, the reward would be the *negative* of the
#     next period's return, minus transaction costs.
#   * The agent's position would need to be updated to reflect the short position
#     (e.g., `pos = -1`).
#   * The logic for handling existing positions would also need to be updated to
#     account for being in a short position.
# The Q-network's output layer would also need to be changed to have three nodes,
# one for each action (long, flat, short).
