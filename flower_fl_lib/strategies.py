# Imports
import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
from .models import NUM_CLIENTS


# Helper functions
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Async FL scenarios
# docs: https://flower.dev/docs/apiref-flwr.html?highlight=server+strategy+fedavg#flwr.server.strategy.FedAvg
STRATEGIES = {
    "best": fl.server.strategy.FedAvg(
        fraction_fit=0.25,
        fraction_evaluate=0.25,
        min_fit_clients=int(NUM_CLIENTS * 0.25),
        min_evaluate_clients=int(NUM_CLIENTS * 0.25),
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    ),
    "worst": fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    ),
    "mid": fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=int(NUM_CLIENTS * 0.5),
        min_evaluate_clients=int(NUM_CLIENTS * 0.5),
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    ),
}
