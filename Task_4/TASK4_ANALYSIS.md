# Task 4 Analysis: A Deeper Dive

This document addresses the excellent questions raised about the methodology and implementation of the reinforcement learning agent in `GroupDTask4.R`.

## 1. Is the 90% efficiency a round-trip efficiency?

No, and this is a critical point. The current implementation applies the 90% efficiency factor **one-way** for both charging and discharging. Let's break down the code:

-   **Charging:** `cost_or_revenue <- -(energy_change / efficiency) * current_price`
-   **Discharging:** `cost_or_revenue <- (energy_change * efficiency) * current_price`

This means if you charge 1 MWh, you pay for `1 / 0.9 = 1.11 MWh`. If you discharge 1 MWh, you get paid for `1 * 0.9 = 0.9 MWh`.

A true **round-trip efficiency** of 90% means that for every 1 MWh you put into the battery, you can only get 0.9 MWh out. This is typically implemented by applying the square root of the efficiency to both charging and discharging (`sqrt(0.9) approx 0.9487`).

**Verdict:** The current implementation is more punitive than a standard round-trip efficiency model. This is a valid modeling choice, but it's important to be aware of the distinction. For this analysis, we will proceed with the current implementation, but it's a key parameter that could be tuned.

## 2. Are the errors from Task 3 present here?

Let's review the teacher's comments from Task 3 and check our `GroupDTask4.R` script.

-   **"condition `if (replay_buffer$idx > warmup_size)` ignores that replay buffer can become full and revert idx back to 1 thus preventing training"**:
    -   **Our code:** `if (replay_buffer$idx > warmup_size || replay_buffer$is_full)`
    -   **Verdict:** **This error is NOT present.** We correctly use `replay_buffer$is_full` to ensure training continues after the buffer has been filled once.

-   **"target net is never compiled"**:
    -   **Our code:** The `target_net` is not compiled.
    -   **Verdict:** **This is not an error.** The target network is only used for making predictions (`predict`), not for training (`train_on_batch`). Therefore, it does not need to be compiled with a loss function and optimizer. Only the online network, which is actively being trained, requires compilation.

-   **"epsilon func does not start at 1 and also does not end at 0.01"**:
    -   **Our code:** `epsilon_start <- 1.0`, `epsilon_final <- 0.01`, and the `epsilon_decay` function correctly uses these.
    -   **Verdict:** **This error is NOT present.** Our epsilon starts at 1.0 and decays towards 0.01 as intended.

## 3. Why 30 days in an episode (`max_steps_ep <- 96 * 30`)?

This is a modeling choice related to creating a meaningful learning experience for the agent in each "run."

-   **Rationale:** An episode should be long enough for the agent to experience a variety of market conditions (e.g., daily and weekly price cycles) but short enough to provide frequent feedback (in the form of a final profit and a reset).
-   **30 Days as a Choice:** A 30-day period (a month) is a common choice in this domain. It's a reasonable proxy for a monthly billing cycle and captures multiple weekly patterns of price volatility. It forces the agent to learn to manage its state-of-charge over a significant period, preventing it from making purely short-sighted decisions.
-   **Could it be different?** Absolutely. A shorter episode (e.g., 7 days) might lead to faster learning of weekly patterns but could miss longer-term trends. A longer episode might be more realistic but would result in less frequent updates and potentially slower learning. The 30-day length is a balanced starting point.

## 4. How much data should we use?

The note "You do not need to use the whole five year period... you should have at least 10000 observations" is a practical hint to balance computational feasibility with model performance.

-   **The Data Scientist's Rationale:** The goal is to provide the agent with a representative sample of the price dynamics it will face in the future.
    -   **Too little data:** The agent might overfit to specific short-term patterns and fail to generalize. For example, training only on a summer month might not prepare it for winter price spikes.
    -   **Too much data:** Training can become computationally expensive and time-consuming. More importantly, very old data might not be representative of current market dynamics (e.g., due to changes in the energy mix, regulations, or demand patterns). This is known as **concept drift**.
-   **Our Approach (`train_years <- 0.02`):**
    -   `0.02 * 365 * 24 * 4 = 700.8` steps. This is very small and likely insufficient for robust learning. It's good for quick debugging, but not for a final model.
    -   The recommendation of **at least 10,000 observations** is a good rule of thumb. This corresponds to `10000 / (24 * 4) = 104` days, or about 3-4 months. This would capture different seasonal patterns.
-   **Recommendation:** A good starting point would be to use one full year of data for training (`train_years = 1`). This would expose the agent to all seasonalities. If that is computationally too expensive, a 6-month period is a reasonable compromise.

## 5. What is the rationale for the train-test split?

In time-series forecasting, we must be extremely careful with how we split our data to avoid **data leakage** (also called lookahead bias).

-   **The Golden Rule:** The test set must always come *after* the training set in time. We are simulating a real-world scenario where we train our model on past data and then deploy it to make decisions on future, unseen data.
-   **Why not a random split?** A random 80:20 split would shuffle the data, meaning the model could be trained on data from the future and tested on data from the past. This would give an unrealistically optimistic evaluation of the model's performance because the model has "seen the future" by learning the patterns from the training data that occurred after the test data points.
-   **Our Approach:** The current script correctly implements a chronological split: `train_prices <- prices[1:train_end_idx]` and `test_prices <- prices[(train_end_idx + 1):test_end_idx]`. This is the correct way to do it.
-   **Choosing the Time Frame:** The choice of the test set's time frame is also important. It should be representative of the period you expect the agent to operate in. For example, testing on a single holiday week might not be representative of a full year's performance. Our current test set is very small (`test_years = 0.01`), which is about 3.65 days. For a more robust evaluation, a longer test period (e.g., 1-3 months) would be better.

## 6. Why does the agent's profit develop like that?

The episode profits you've shared (`-1275.78`, `-1199.59`, `-984.51`, etc.) show a classic reinforcement learning pattern, especially in the early stages.

-   **Initial Exploration:** In the first few episodes, epsilon is high, meaning the agent is taking mostly random actions (exploring). It doesn't yet have a "policy" or strategy. It's essentially buying and selling at random, and with transaction costs (efficiency loss and degradation), this leads to a net loss.
-   **Gradual Learning (Exploitation):** As training progresses, epsilon decays, and the agent starts to exploit what it has learned. It begins to associate low prices with the "charge" action and high prices with the "discharge" action.
-   **Fluctuations:** The profit won't increase smoothly. You'll see ups and downs. An agent might find a suboptimal local minimum (e.g., a simple strategy that avoids big losses but also misses big gains) and then, through further exploration, discover a better strategy, causing the profit to jump.
-   **The Goal:** The goal is that, over many episodes, the average profit will trend upwards as the agent's Q-function becomes a more accurate predictor of the long-term value of each action in each state. The fact that your final profit on the test set (20.55) is positive and significantly better than the random agent (-485.89) is a strong indication that **learning is happening**.

---

Next, I will modify the R script to implement the enhanced evaluation and logging as we discussed.

---

## 7. In-depth Evaluation and Interpretation

To properly evaluate our agents, we need to go beyond the final profit figure. The enhanced script now logs the agent's actions, SoC, and the market price at every step of the test period. This data is saved in `evaluation_results.csv` and visualized in `profit_plot.png` and `agent_behavior_plot.png`.

### How to Interpret the Results:

1.  **`profit_plot.png`:** This plot is your primary tool for comparing the overall performance of the different policies (DQN, DDQN, Random, Heuristic).
    *   **Look for:** Which agent achieves the highest cumulative profit? Is the profit growth steady or volatile? Do the RL agents (DQN/DDQN) consistently outperform the benchmarks? A steeper upward slope indicates a more profitable strategy.

2.  **`agent_behavior_plot.png`:** This is where you'll find the "why" behind an agent's performance. The plot shows the agent's decisions (charge/discharge) against the backdrop of the electricity price and the battery's state of charge.
    *   **A "good" agent should exhibit a clear strategy:**
        *   **Charging (Green Dots):** These should predominantly occur when the price (black line) is low.
        *   **Discharging (Red Dots):** These should predominantly occur when the price is high.
        *   **State of Charge (Blue Dashed Line):** The SoC should decrease during high-price periods (as the agent sells energy) and increase during low-price periods (as it buys energy).
    *   **Compare DQN vs. DDQN:** By looking at the two facets of this plot, you can directly compare the trading strategies of the two agents. Does one seem to capture price swings more effectively? Is one more conservative than the other?

3.  **`evaluation_results.csv`:** This file contains the raw data for your own deep-dive analysis. You can use it to calculate specific metrics, such as:
    *   Average purchase price vs. average selling price.
    *   Number of charge/discharge cycles (to analyze degradation).
    *   The agent's behavior during extreme price events (e.g., negative prices or very high spikes).

By combining the insights from these three sources, you can build a comprehensive narrative that answers the key question: **Why is your agent better or worse in terms of overall profit?** You can point to specific instances in the behavior plot to justify your conclusions about the agent's learned strategy.
