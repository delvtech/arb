# %%
from utils.check_env import running_interactive
from collections import namedtuple
import os
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt

config = wandb.config if "WANDB_RUN_ID" in os.environ else {}

# %%
# setup
Normal = namedtuple("Normal", ["mean", "std"])
num_blocks = 1000 if "num_blocks" not in config else int(config["num_blocks"])
rate = Normal(0.035, 0.02)
trade = Normal(0.002, 0.001)
randseed = 0 if "randseed" not in config else int(config["randseed"])

# %%
# run shit
# fees = 0.005 if "fees" not in config else float(config["fees"])
rate_histories = []
records = []
min_fees = 0.0001
max_fees = 0.01
increment = 0.0001
for fees in np.arange(min_fees, max_fees + increment, increment):
    for epoch in range(100):
        rng = np.random.default_rng(seed=epoch)
        rate_history = [rate.mean]
        volume = 0
        for _ in range(1, num_blocks):
            # every trader has a different target rate
            target = rng.normal(rate.mean, rate.std)
            # trade if rate is outside of dead zone around their target
            diff_to_true_mean = abs(rate_history[-1] - target)
            if diff_to_true_mean > fees:
                tradesize = min(max(0, rng.normal(trade.mean, trade.std)), diff_to_true_mean)
                direction = -1 if target < rate_history[-1] else 1
                new_rate = rate_history[-1] + direction * tradesize
                rate_history.append(new_rate)
                volume += abs(rate_history[-1] - rate_history[-2])
        money_made = volume * fees
        # print(f"fees={fees}, money_made={money_made}, volume={volume}")
        records.append((fees, epoch, money_made, volume))
        rate_histories.append(rate_history)

# %%
# analysis
if running_interactive():
    rh_df = pd.DataFrame(rate_histories)
    plt.plot(rh_df.values.T, alpha=0.01, color="black")
    plt.show()
    results = pd.DataFrame(records, columns=["fees", "epoch", "money_made", "volume"])
    display(results)

    sns.relplot(x="fees", y="money_made", kind="line", data=results, color="blue")
    plt.ylabel("money_made", color="blue")
    plt.tick_params(axis='y', color="blue", labelcolor="blue")
    plt.gca().spines['left'].set_color("blue")
    plt.title("money_made vs. fees")
    plt.show()

    sns.relplot(x="fees", y="volume", kind="line", data=results, color="red")
    plt.ylabel("volume", color="red")
    plt.tick_params(axis='y', color='red', labelcolor='red')
    plt.gca().spines['left'].set_color("red")
    plt.title("volume vs. fees")
    plt.show()

# %%
