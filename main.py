#! ./.venv/bin/python3
# Imports
import fire
import flwr as fl
from flower_fl_lib import client_fn, FL_STRATEGIES, FL_NUM_CLIENTS

# h.metrics_distributed, h.losses_distributed

def run_simulation(strat: str = "best", num_rounds: int = 5) -> fl.server.history.History:
    """
    Run Federated Learning Simulation\n
    strat: "best", "worst", or "mid"
    """
    strategy = FL_STRATEGIES[strat]

    return fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=FL_NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    fire.Fire(run_simulation)
