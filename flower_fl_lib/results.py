import flwr as fl
from datetime import datetime
import csv
from pathlib import Path
from .models import Net, FlowerClient, DEVICE, NUM_CLIENTS, trainloaders, valloaders
from .strategies import STRATEGIES


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)


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
    strategy = STRATEGIES[strat]
    return fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def async_fl_simulation(strat: str = "best", num_rounds: int = 5) -> None:
    time_stamp = datetime.now()
    data = run_simulation(strat=strat, num_rounds=num_rounds)
    save_simulation(results=data, time_stamp=time_stamp)
