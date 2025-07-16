library(keras)
library(quantmod)

## HYPER-PARAMETERS  -----------------------------------------------------
symbol            <- "SPY"
start_date        <- "2015-01-01"
window_size       <- 10            # look-back bars → state vector length
state_dim <- window_size + 1           # returns window  +  current position
hidden_units      <- 32
learning_rate     <- 1e-3
gamma             <- 0.99          # discount
episodes          <- 30           # +/- depending on patience

batch_size        <- 32
replay_capacity   <- 5000
warmup_mem        <- 200           # no training until buffer this large
target_sync_freq  <- 10            # episodes

eps_start         <- 1.0           # ε-greedy schedule
eps_final         <- 0.05

trade_cost <- 0.0005      


## LOAD DATA  ------------------------------------------------------------
getSymbols(symbol, from = start_date, auto.assign = TRUE)
price <- Cl(get(symbol))
rets  <- na.omit(dailyReturn(price))           # % daily returns

# training and test split 
train_size <- floor(0.8 * length(rets))
rets_train <- rets[1:train_size]
rets_test  <- rets[(train_size + 1):length(rets)]

rets <- rets_train
n_obs <- nrow(rets)
max_steps_ep      <- n_obs - window_size - 1           # guard against infinite loops

## STATE: # t is  (window_size : n_obs-1) and current position (0/1) as the last feature
make_state <- function(t, pos) {
  c(as.numeric(rets[(t - window_size + 1):t]), pos)
}


## ENVIRONMENT HELPERS  --------------------------------------------------
env_reset <- function() list(t = window_size, pos = 0L,
                             equity = 1.0, done = FALSE)

env_step <- function(env, action) {
  # action: 0 = stay/turn flat, 1 = stay/turn long
  t2        <- env$t + 1
  switchfee <- if (action != env$pos) trade_cost else 0
  pos_next  <- action
  
  ret  <- if (pos_next == 1) as.numeric(rets[t2]) else 0
  rew  <- ret - switchfee                     # reward *now*
  
  list(
    next_env = list(t = t2, pos = pos_next,
                    equity = env$equity * (1 + rew),
                    done = (t2 >= n_obs - 1)),
    reward   = rew,
    obs      = make_state(t2, pos_next)
  )
}


## BUILD Q-NETWORK + TARGET NETWORK  -------------------------------------
build_qnet <- function() {
  input  <- layer_input(shape = state_dim)   
  hidden <- input %>% layer_dense(hidden_units, activation = "relu")
  output <- hidden %>% layer_dense(2, activation = "linear")
  keras_model(input, output) %>% 
    compile(optimizer = optimizer_adam(learning_rate), loss = "mse")
}

qnet        <- build_qnet()
target_net  <- build_qnet()               # initial weights identical
target_net$set_weights(qnet$get_weights())

## REPLAY BUFFER  --------------------------------------------------------
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

## TRAINING LOOP  --------------------------------------------------------
eps_decay_rate <- log(eps_start / eps_final) / (episodes - 1)

epsilon <- function(ep) {
  eps_start * exp(-eps_decay_rate * (ep-1))
}

for (ep in 1:episodes) {
  env   <- env_reset()
  state <- make_state(env$t, env$pos)
  eps   <- epsilon(ep)
  ep_reward <- 0
  for (step in 1:max_steps_ep) {
    # choose action
    if (runif(1) < eps)
      action <- sample(0:1, 1)                          # explore
    else
      action <- which.max(predict(qnet, matrix(state, nrow = 1), verbose = 0)) - 1   # exploit
    
    # interact
    res <- env_step(env, action)
    store_transition(state, action, res$reward, res$obs, res$next_env$done)
    
    state <- res$obs
    env   <- res$next_env
    ep_reward <- ep_reward + res$reward
    if (env$done) break
    
    # learn
    if ( (replay$full || replay$idx > warmup_mem) ) {
      batch <- sample_batch(batch_size)
      q_next <- predict(target_net, batch$S2, verbose = 0)
      q_max  <- apply(q_next, 1, max)
      y      <- batch$R + (1 - batch$D) * gamma * q_max
      y_keras <- predict(qnet, batch$S, verbose = 0) # this is only needed to generate q values for actions not taken
      y_keras[cbind(seq_len(batch_size), batch$A + 1)] <- y # crucial mostly for keras, we set the y value only for taken actions to get their loss (y-q)^2 and leave the rest to the original q-values, essentially setting their loss to be 0 (q-q)^2
      qnet %>% train_on_batch(batch$S, y_keras) # y_keras is the artificial "true" target, so we train the qnet to predict these values
    }
  }
  
  # target network sync
  if (ep %% target_sync_freq == 0)
    target_net$set_weights(qnet$get_weights())
  
  cat(sprintf("Episode %3d | ε = %.3f | steps = %3d | reward = %.4f | equity = %.3f\n",
              ep, eps, step, ep_reward, env$equity))
}

## Quick benchmark in-sample  --------------------------------------------------------
# play one deterministic episode (ε=0) and plot equity vs buy&hold
env   <- env_reset(); state <- make_state(env$t, env$pos)
eq_curve <- numeric()
decision <- integer() # not part of the task, I was just interested
repeat {
  action <- which.max(predict(qnet, matrix(state, nrow = 1), verbose = 0)) - 1L
  res    <- env_step(env, action)
  eq_curve <- c(eq_curve, res$next_env$equity)
  if (res$next_env$done) break
  env   <- res$next_env
  state <- res$obs
  decision <- c(decision, action) # not part of the task, I was just interested
}
bh_curve <- cumprod(1 + rets[(window_size+1):n_obs])

plot(coredata(eq_curve), type = "l", lwd = 2,
     main = "Equity Curve vs Buy-&-Hold (test run)",
     ylab = "Wealth", xlab = "Trading step",ylim=c(0,4))
lines(coredata(bh_curve), col = 2, lwd = 2, lty = 2)
legend("topleft", c("Agent", "Buy-&-Hold"), lwd = 2, col = c(1,2), lty = 1:2)
abline(v = which(decision == 0), col = "grey60", lty = 3) # not part of the task, I was just interested
