# Imports
from .models import Net, FlowerClient, DEVICE, NUM_CLIENTS, trainloaders, valloaders
from .strategies import STRATEGIES

# Constants

# CLASSES = (
#     "plane",
#     "car",
#     "bird",
#     "cat",
#     "deer",
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck",
# )

FL_NUM_CLIENTS = NUM_CLIENTS
FL_STRATEGIES = STRATEGIES


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


# Create FedAvg strategy
# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=1.0,
#     fraction_evaluate=0.5,
#     min_fit_clients=10,
#     min_evaluate_clients=5,
#     min_available_clients=10,
#     evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
# )
