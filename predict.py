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
num_blocks = 1000 if "num_blocks" not in config else int(config["num_blocks"])
blocks = range(10)
fee_constant_range = [0.1]
epochs = range(1)
blocks = range(num_blocks)
fee_constant_range = np.arange(min_fees, max_fees + increment, increment)
epochs = range(10)
for epoch in epochs:
    for fee_constant in fee_constant_range:
        rng = np.random.default_rng(seed=epoch)
        rate_history = [rate.mean]
        target_history = [np.nan]
        volume = 0
        lp_profit = 0
        trader_profit = 0
        true_rate = rng.normal(rate.mean, rate.std)
        for block in blocks:
            # every trader has a different target rate
            target = rng.normal(rate.mean, rate.std)
            target_history.append(target)
            # trade if rate is outside of dead zone around their target
            diff_to_target = abs(rate_history[-1] - target)
            # "fees_in_base" = market_rate * fee_rate
            #        = 3.5% * 10%
            #        = 0.35%
            # if we were using yieldspace
            #        = 1/p - 1 * 10%
            #        p is a function of rate
            #        fees_in_base are a function of price
            #        fees_in_base = f(p(r))
            # what does that mean?
            # it models the dead zone by measuring half of it
            # in this example, the full dead zone is:
            #   dead zone = 3.5% +/- 0.35%
            #             = 3.15 - 3.85%
            # what does this mean?
            # if your target is within this range, you will never trade
            # if your target is outside of this range, then you consider slippage
            # if your target is 3%
            # absent fees, you'd move the market to 3%
            # go long at 3%, what do you get? your profit is only 3% * (1-fees)
            dead_zone = (rate_history[-1] * (1-fee_constant), rate_history[-1] * (1+fee_constant))
            if debug:
                print(f"{dead_zone=}")
            # gonna trade if outside of dead zone
            gonna_trade = target > max(dead_zone) or target < min(dead_zone)
            if debug:
                print(f"{target=} {diff_to_target=} {gonna_trade=}")
            if gonna_trade:
                direction = -1 if target < rate_history[-1] else 1
                if direction == 1:  # want rate to go up
                    # we stop at the lopwer end of the dead zone
                    endpoint = target * (1-fee_constant)
                else:  # want rate to go down
                    # we stop at the upper end of the dead zone
                    endpoint = target * (1+fee_constant)
                tradesize = abs(rate_history[-1] - endpoint)
                if debug:
                    print(f"{endpoint=} {tradesize=}")
                new_rate = rate_history[-1] + direction * tradesize
                rate_history.append(new_rate)
                rate_change = abs(rate_history[-1] - rate_history[-2])
                # rate change is a function of base traded
                volume += rate_change
                trade_fee = rate_history[-1] * fee_constant * tradesize
                lp_profit += trade_fee
                # missing distribution of trader returns (some profitable, some not)
                # add that in by giving them prediction accuracy
                trader_profit += rate_change - trade_fee
                if debug:
                    print(f"{block=} {rate_change=} {rate_history[-1] * fee_constant * tradesize=}")
                trades += 1
            else:
                rate_history.append(rate_history[-1])
        records.append((fee_constant, epoch, lp_profit, volume))
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
    plt.plot(rh_df.values.T[:,0], alpha=1, color="blue", label="market rate", linewidth=0.5)
    errors = [t*fee_constant for t in target_histories[0]]
    plt.scatter(np.arange(len(target_histories[0])), target_histories[0], color="red", s=1, label="target rate")
    # plt.errorbar(np.arange(len(target_histories[0])), target_histories[0], color="red", label="target rate", yerr=errors, elinewidth=0.5, capsize=0.5)
    plt.set_xlabel("block number")
    plt.set_ylabel("fixed rate (%)")
    # plt.xlim(0, 100)
    plt.set_xlim(0, 100)
    plt.legend()

# %%
# inspect multiple trade histories in one plot
if RUNNING_INTERACTIVE:
    fee_choices = [0, 1, 10]
    num_choices = len(fee_choices)
    fig, ax = plt.subplots(num_choices, 1, figsize=(8, 12), sharex=True, tight_layout=True)
    for idx, which_fee in enumerate(fee_choices):
        ax[idx].plot(rh_df.values.T[:,which_fee], alpha=1, color="blue", label="market rate", linewidth=0.5)
        errors = [t*fee_constant for t in target_histories[0]]
        ax[idx].scatter(np.arange(len(target_histories[0])), target_histories[0], color="red", s=1, label="target rate")
        # plt.errorbar(np.arange(len(target_histories[0])), target_histories[0], color="red", label="target rate", yerr=errors, elinewidth=0.5, capsize=0.5)
        if idx == num_choices-1:
            ax[idx].set_xlabel("block number")
        ax[idx].set_ylabel("fixed rate (%)")
        # plt.xlim(0, 100)
        ax[idx].set_xlim(0, 100)
        ax[idx].set_title(f"fees={which_fee/100:.0%}")
        ax[idx].legend()

# %%
# inspect multiple trade histories separate plots
if RUNNING_INTERACTIVE:
    fee_choices = [0, 2, 10]
    num_choices = len(fee_choices)
    # fig, ax = plt.subplots(num_choices, 1, figsize=(8, 12), sharex=True, tight_layout=True)
    for which_fee in fee_choices:
        plt.figure(figsize=(8, 4))
        plt.plot(rh_df.values.T[:,which_fee], alpha=1, color="blue", label="market rate", linewidth=0.5)
        errors = [t*fee_constant for t in target_histories[0]]
        plt.scatter(np.arange(len(target_histories[0])), target_histories[0], color="red", s=1, label="target rate")
        # plt.errorbar(np.arange(len(target_histories[0])), target_histories[0], color="red", label="target rate", yerr=errors, elinewidth=0.5, capsize=0.5)
        plt.xlabel("block number")
        plt.ylabel("fixed rate (%)")
        # plt.xlim(0, 100)
        plt.xlim(0, 100)
        plt.title(f"fees={which_fee/100:.0%}")
        plt.legend()

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
    ax2 = ax.twinx()
    sns.lineplot(x="fee_rate", y="lp_profit", data=results, color="blue", ax=ax2, label="lp profit", errorbar="pi")
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
# plot together
if RUNNING_INTERACTIVE:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax2 = ax.twinx()
    sns.lineplot(x="fee_rate", y="lp_profit", data=results, color="blue", ax=ax2, label="lp profit", errorbar="pi")
    sns.lineplot(x="fee_rate", y="volume", data=results, color="red", ax=ax, label="volume", errorbar="pi")
    plt.xlim([0, 0.4])
    ax.set_xlabel("percentage of trade captured by fees")
    plt.ylabel("profit (% points)")
    max_lp_profit = np.round(results['fee_rate'][results["lp_profit"].idxmax()], 2)
    # plt.title(f"max lp profit at fees={max_lp_profit}")
    plt.title(f"LP profit and trade volume vs. fee multiplier")
    legend_lines = [ax2.lines[0], ax.lines[0]]
    legend_labels = [ax2.get_lines()[0].get_label(), ax.get_lines()[0].get_label()]
    plt.legend(legend_lines, legend_labels);

# %%