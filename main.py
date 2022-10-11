# Imports
import flwr as fl
from flower_fl_lib import client_fn, FL_STRATEGIES, FL_NUM_CLIENTS

if __name__ == "__main__":
    strategy = FL_STRATEGIES["best"]
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=FL_NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
