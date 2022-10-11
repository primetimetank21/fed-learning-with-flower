#! ./.venv/bin/python3
# Imports
import fire
from flower_fl_lib import async_federated_learning_training


if __name__ == "__main__":
    fire.Fire(async_federated_learning_training)
