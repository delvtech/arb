# %%
from utils.check_env import running_interactive

# %%
from collections import namedtuple
import numpy as np
import wandb
from matplotlib import pyplot as plt

# num_blocks = 10_000
Normal = namedtuple("Normal", ["mean", "std"])
num_blocks = 1000 if "num_blocks" not in wandb.config else int(wandb.config["num_blocks"])
rate = Normal(0.035, 0.02)
trade = Normal(0.002, 0.001)
rate_history = [rate.mean]
randseed = 0 if "randseed" not in wandb.config else int(wandb.config["randseed"])
rng = np.random.default_rng(seed=randseed)
fees = 0.005 if "fees" not in wandb.config else float(wandb.config["fees"])
print(f"{randseed=}")
print(f"{fees=}")
lvr_history = [0]
volume = 0
# create price history
for _ in range(1, num_blocks):
    target = rng.normal(rate.mean, rate.std)

    # dumb trade on every block
    tradesize = max(0, rng.normal(trade.mean, trade.std))
    direction = rng.choice([-1, 1])
    new_rate = rate_history[-1] + direction * tradesize
    rate_history.append(new_rate)
    lvr_history.append(lvr_history[-1])
    volume += abs(rate_history[-1] - rate_history[-2])

    # optional smart trade if outside of dead zone
    diff_to_true_mean = abs(rate_history[-1] - rate.mean)
    if diff_to_true_mean > fees:
        tradesize = min(max(0, rng.normal(trade.mean, trade.std)), diff_to_true_mean)
        direction = -1 if target < rate_history[-1] else 1
        new_rate = rate_history[-1] + direction * tradesize
        rate_history.append(new_rate)
        lvr_history.append(lvr_history[-1] + abs(rate_history[-1] - rate_history[-2]))
        volume += abs(rate_history[-1] - rate_history[-2])
    # print(f"len(rate_history): {len(rate_history)}, len(lvr_history): {len(lvr_history)}")
money_made = volume * fees
print(f"money_made={money_made}")
print(f"lvr={lvr_history[-1]}")
print(f"volume={volume}")
if running_interactive():
    x = range(len(rate_history))
    plt.plot(rate_history)
    plt.title(f"fees={fees}, volume={volume}")
    ax2 = plt.gca().twinx()
    ax2.plot(lvr_history, color="red")
    ax2.set_ylabel("LVR", color="red")
    ax2.spines['right'].set_color("red")

# %%