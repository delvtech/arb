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
rate = Normal(0.035, 0.005)  # mean of 3.5% and std of 0.5%
randseed = 0 if "randseed" not in config else int(config["randseed"])

# %%
# run simulation
rate_histories = []
target_histories = []
records = []
min_fees = 0.0001
max_fees = 1
increment = 0.01
trades = 0
start_time = time.time()
debug = False
for fee_rate in np.arange(min_fees, max_fees + increment, increment):
# for fee_rate in [0.2]:
    for epoch in range(100):
    # for epoch in range(1):
        rng = np.random.default_rng(seed=epoch)
        rate_history = [rate.mean]
        target_history = [np.nan]
        volume = 0
        lp_profit = 0
        for block in range(1, num_blocks):
            # every trader has a different target rate
            target = rng.normal(rate.mean, rate.std)
            target_history.append(target)
            # trade if rate is outside of dead zone around their target
            diff_to_target = abs(rate_history[-1] - target)
            fees = rate_history[-1] * fee_rate
            if debug:
                print(f"{target=} {diff_to_target=} {fees=}")
            if diff_to_target > fees:
                direction = -1 if target < rate_history[-1] else 1
                tradesize = max(0, diff_to_target-fees)
                new_rate = rate_history[-1] + direction * tradesize
                rate_history.append(new_rate)
                rate_change = abs(rate_history[-1] - rate_history[-2])
                volume += rate_change
                lp_profit += fees
                if debug:
                    print(f"{block=} {rate_change=} {fees=}")
                trades += 1
            else:
                rate_history.append(rate_history[-1])
        records.append((fee_rate, epoch, lp_profit, volume))
        rate_histories.append(rate_history)
        target_histories.append(target_history)
        if debug:
            plt.plot(rate_history)
print(f"simulated {trades=} trades in {time.time() - start_time} seconds")

# %%
# create dataframe
if RUNNING_INTERACTIVE:
    rh_df = pd.DataFrame(rate_histories)
    target_df = pd.DataFrame(target_histories)
    results = pd.DataFrame(records, columns=["fee_rate", "epoch", "lp_profit", "volume"])
    results["profit_and_volume"] = results["lp_profit"] + results["volume"] * 0.5
    # display(results)

# %%
# inspect one trade history
if RUNNING_INTERACTIVE:
    plt.figure(figsize=(8, 4))
    plt.plot(rh_df.values.T[:,0], alpha=1, color="blue", label="market rate", linewidth=0.5)
    errors = [t*fee_rate for t in target_histories[0]]
    plt.scatter(np.arange(len(target_histories[0])), target_histories[0], color="red", s=1, label="target rate")
    # plt.errorbar(np.arange(len(target_histories[0])), target_histories[0], color="red", label="target rate", yerr=errors, elinewidth=0.5, capsize=0.5)
    plt.xlabel("block number")
    plt.ylabel("fixed rate (%)")
    plt.xlim(0, 100)
    plt.legend()
    plt.show()

# %%
# plot all rate histories
if RUNNING_INTERACTIVE:
    plt.figure(figsize=(8, 4))
    plt.plot(rh_df.values.T, alpha=0.005, color="black")
    plt.xlabel("block number")
    plt.ylabel("fixed rate (%)")
    plt.title("all rate paths")
    plt.show()  

# %%
# plot together
if RUNNING_INTERACTIVE:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.lineplot(x="fee_rate", y="lp_profit", data=results, color="blue", ax=ax, label="lp profit", errorbar="pi")
    sns.lineplot(x="fee_rate", y="volume", data=results, color="red", ax=ax, label="volume", errorbar="pi")
    sns.lineplot(x="fee_rate", y="profit_and_volume", data=results, color="green", ax=ax, label="volume/2 + lp profit", errorbar="pi")
    plt.xlim([0, 0.2])
    plt.xlabel("fees")
    plt.ylabel("profit (% points)")
    max_profit_and_volume = np.round(results['fee_rate'][results["profit_and_volume"].idxmax()], 2)
    max_lp_profit = np.round(results['fee_rate'][results["lp_profit"].idxmax()], 2)
    plt.title(f"max profit_and_volume at fees={max_profit_and_volume}, lp_profit at fees={max_lp_profit}")
    plt.legend();

# %%
