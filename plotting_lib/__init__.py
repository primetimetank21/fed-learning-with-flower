import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def plot_line(col: pd.core.series.Series, filename: str = "./line_plot.png"):
    sns.lineplot(col)
    plt.savefig(filename)
    plt.clf()


if __name__ == "__main__":
    scenario_dir_names = [name for name in Path("./results").iterdir()]
    for scenario in scenario_dir_names:
        scenario_files = [file_name for file_name in scenario.iterdir()]
        for file_name in scenario_files:
            print(file_name)
            df = pd.read_csv(file_name)
            plot_name = str(file_name).rsplit("/", maxsplit=1)[-1].replace(".csv", "")
            for feature in df.columns:
                plot_line(df[feature], f"{plot_name}-{feature}.png")
            break
