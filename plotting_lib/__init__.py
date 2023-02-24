from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def plot_line(col: pd.core.series.Series, filename: str = "./line_plot.pdf"):
    sns.lineplot(col)
    plt.xlabel("epochs")
    plt.savefig(filename)
    plt.clf()


def create_graphs():
    scenario_dir_names = list(Path("./results").iterdir())
    Path("./results_metrics").mkdir(parents=True, exist_ok=True)

    for scenario in scenario_dir_names:
        scenario_files = list(scenario.iterdir())
        scenario_dir_path = Path(f"./results_metrics/{scenario.name}")
        scenario_dir_path.mkdir(parents=True, exist_ok=True)

        for file_name in scenario_files:
            df = pd.read_csv(file_name)
            plot_name = str(file_name).rsplit("/", maxsplit=1)[-1].replace(".csv", "")
            Path(f"./{scenario_dir_path}/{plot_name}").mkdir(
                parents=True, exist_ok=True
            )
            for feature in df.columns:
                plot_line(
                    df[feature], f"./{scenario_dir_path}/{plot_name}/{feature}.pdf"
                )


def create_comparison_graph(dir_name: str, plot_name: str):
    files = list(Path(dir_name).iterdir())
    scenario_data = [pd.read_csv(file) for file in files]
    scenario_names = ["best", "middle", "worst"]
    file_path = Path(f"{dir_name}_graphs")
    file_path.mkdir(parents=True, exist_ok=True)
    if plot_name[-4:] != ".pdf":
        plot_name += ".pdf"

    for feature in scenario_data[0].columns:
        for i, sd in enumerate(scenario_data):
            sns.lineplot(sd[feature], label=scenario_names[i].title())

        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(feature.title())
        plt.savefig(f"{file_path}/{feature}_with_{plot_name}")
        plt.clf()


# create_comparison_graph(["./250_epochs/best_250_epochs.csv","./250_epochs/mid_250_epochs.csv","./250_epochs/worst_250_epochs.csv"],
# "250_epochs.pdf")
