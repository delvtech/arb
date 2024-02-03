# %%
from utils.check_env import running_interactive
from collections import namedtuple
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt

RUNNING_INTERACTIVE = running_interactive()

config = wandb.config if "WANDB_RUN_ID" in os.environ else {}

# %%
# setup
Normal = namedtuple("Normal", ["mean", "std"])
num_blocks = 1000 if "num_blocks" not in config else int(config["num_blocks"])
rate = Normal(0.035, 0.02)
randseed = 0 if "randseed" not in config else int(config["randseed"])

# %%
# run shit
# fees = 0.005 if "fees" not in config else float(config["fees"])
rate_histories = []
records = []
min_fees = 0.0001
max_fees = 1
increment = 0.01
trades = 0
start_time = time.time()
debug = False
for fee_rate in np.arange(min_fees, max_fees + increment, increment):
# for fee_rate in [max_fees]:
    for epoch in range(10):
    # for epoch in range(1):
        rng = np.random.default_rng(seed=epoch)
        rate_history = [rate.mean]
        volume = 0
        trader_profit = 0
        lp_profit = 0
        for _ in range(1, num_blocks):
            # every trader has a different target rate
            target = rng.normal(rate.mean, rate.std)
            # trade if rate is outside of dead zone around their target
            diff_to_target = abs(rate_history[-1] - target)
            fees = rate_history[-1] * fee_rate
            if debug:
                print(f"{target=} {diff_to_target=} {fees=}")
            if diff_to_target > fees:
                direction = -1 if target < rate_history[-1] else 1
                new_rate = rate_history[-1] + direction * diff_to_target
                rate_history.append(new_rate)
                rate_change = abs(rate_history[-1] - rate_history[-2])
                volume += rate_change
                trader_profit += rate_change - fees
                lp_profit += fees
                if debug:
                    print(f"{rate_change=} {fees=} {trader_profit=}")
                trades += 1
        records.append((fee_rate, epoch, trader_profit, lp_profit, volume))
        rate_histories.append(rate_history)
        if debug:
            plt.plot(rate_history)
print(f"simulated {trades=} trades in {time.time() - start_time} seconds")

# %%
# create dataframe
if RUNNING_INTERACTIVE:
    rh_df = pd.DataFrame(rate_histories)
    results = pd.DataFrame(records, columns=["fee_rate", "epoch", "trader_profit", "lp_profit", "volume"])
    results["happiness"] = results["lp_profit"] + results["trader_profit"] * 0.5
    display(results)

# %%
# rate history
if RUNNING_INTERACTIVE:
    plt.plot(rh_df.values.T, alpha=0.005, color="black")
    plt.show()  

# %%
# subplots
fig, ax = plt.subplots(3, 1, figsize=(10, 10))

# lp profit
if RUNNING_INTERACTIVE:
    sns.lineplot(x="fee_rate", y="lp_profit", data=results, color="blue", ax=ax[0], label="lp profit")
    plt.ylabel("profit", color="blue")
    ax[0].tick_params(axis='y', color="blue", labelcolor="blue")
    ax[0].spines['left'].set_color("blue")
    ax[0].legend()

# trader profit
if RUNNING_INTERACTIVE:
    sns.lineplot(x="fee_rate", y="trader_profit", data=results, color="orange", ax=ax[1], label="trader profit")
    plt.ylabel("profit", color="orange")
    ax[1].tick_params(axis='y', color="orange", labelcolor="orange")
    ax[1].spines['left'].set_color("orange")
    ax[1].legend()

# volume
if RUNNING_INTERACTIVE:
    sns.lineplot(x="fee_rate", y="volume", data=results, color="red", ax=ax[2], label="volume")
    plt.ylabel("volume", color="red")
    ax[2].tick_params(axis='y', color='red', labelcolor='red')
    ax[2].spines['left'].set_color("red")
    ax[2].legend()

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.lineplot(x="fee_rate", y="lp_profit", data=results, color="blue", ax=ax, label="lp profit")
sns.lineplot(x="fee_rate", y="trader_profit", data=results, color="orange", ax=ax, label="trader profit")
sns.lineplot(x="fee_rate", y="volume", data=results, color="red", ax=ax, label="volume")
sns.lineplot(x="fee_rate", y="happiness", data=results, color="green", ax=ax, label="happiness")
plt.xlabel("fees")
plt.ylabel("profit (% points)")
# find index where happiness is max
max_index = results["happiness"].idxmax()
plt.title(f"max happiness at fees={results['fee_rate'][max_index]}")
plt.legend();

# %%
