#! ./.venv/bin/python3
# Imports
from datetime import datetime
import csv
from pathlib import Path
import fire
import flwr as fl
from flower_fl_lib import client_fn, FL_STRATEGIES, FL_NUM_CLIENTS


def save_simulation(results: fl.server.history.History, time_stamp: datetime) -> None:
    # Reformat data
    rows = []
    for loss, acc in zip(
        results.losses_distributed, results.metrics_distributed["accuracy"]
    ):
        if loss[0] != acc[0]:
            raise Exception("Mismatched data")

        row = {"loss": loss[1], "accuracy": acc[1]}
        rows.append(row)

    field_names = list(rows[0].keys())

    # Write to file
    file_time = time_stamp.strftime("%m-%d-%Y_at_%H-%M-%S")
    file_name = f"async_fl_run_{file_time}.csv"
    file_path = Path(f"results/{file_name}")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def run_simulation(
    strat: str = "best", num_rounds: int = 5
) -> fl.server.history.History:
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


def async_fl_simulation(strat: str = "best", num_rounds: int = 5) -> None:
    time_stamp = datetime.now()
    data = run_simulation(strat=strat, num_rounds=num_rounds)
    save_simulation(results=data, time_stamp=time_stamp)


if __name__ == "__main__":
    fire.Fire(async_fl_simulation)
