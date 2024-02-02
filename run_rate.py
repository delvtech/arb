import os
import wandb
import importlib

wandb.login()

# set up parameters
parameters = {
    "randseed": {"distribution": "int_uniform", "min": 0, "max": 10_000},
    "fees": {"min": 0.0001, "max": 0.1},
}
sweep_configuration = {
    "name": "deeadzone",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "money_made"},
    "parameters": parameters
}

# # load sweep_id from sweep_id.env
if os.path.exists("sweep_id.env"):
    with open("sweep_id.env", "r", encoding="utf-8") as f:
        sweep_id = f.read().strip()
        print(f"{sweep_id=}")
else:
    # set up sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="deeadzone")
    print(f"{sweep_id=}")
    # save sweep_id to sweep_id.env
    with open("sweep_id.env", "w", encoding="utf-8") as f:
        f.write(sweep_id)

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="deeadzone")

def main():
    wandb.init()
    config = wandb.config
    money_made, volume, lvr, lvr_history, rate_history = run_experiment(config)
    wandb.log({"money_made": money_made})
    wandb.log({"volume": volume})
    wandb.log({"lvr": lvr})
    wandb.log({"lvr_history": lvr_history})
    wandb.log({"rate_history": rate_history})

def run_experiment(config):
    config = wandb.config
    print(f"{config=}")
    
    # Dynamically import the 'rate' module
    rate = importlib.import_module("rate")

    # Reload the module to ensure it's the latest version
    importlib.reload(rate)

    # Access variables directly from the module object
    return rate.money_made, rate.volume, rate.lvr_history[-1], rate.lvr_history, rate.rate_history

# run a wandb agent
wandb.agent(sweep_id, function=main, count=500)
