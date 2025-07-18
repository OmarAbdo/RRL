# Reinforcement Learning for Battery Dispatch: A Capstone Project

## 1. Introduction

This report details the development and evaluation of a Reinforcement Learning (RL) agent designed to operate a 1 MWh battery in the Austrian electricity market. The primary objective is to maximize profit by arbitraging intraday price swings.

## 2. Baseline Agent (T1)

This section describes the baseline Deep Q-Network (DQN) agent developed to trade on historical price data.

## 3. Benchmarks (T2)

The performance of the DQN agent is benchmarked against two simpler policies: a random policy and a heuristic policy.

## 4. Evaluation (T3)

This section provides an in-depth evaluation of the agent's performance on the test set, comparing its trading decisions and profitability against the benchmarks.

## 5. Mandatory Enhancement (T4)

To improve upon the baseline model, a Dueling Double DQN with Prioritized Experience Replay (PER) was implemented. This section details the enhancement and its impact on performance.

## 6. Stored Weights (T5)

The trained network parameters for both the baseline and enhanced models have been saved to external files (`model_weights_baseline.h5` and `model_weights_per.h5`).

## 7. Conclusion

This report concludes with a summary of the project's findings and potential avenues for future research.
