import seaborn as sns
import pandas as pd
from pathlib import Path

def main_24():
    """
    Plot the data visualisation of the section 2.4 of hw3 :

    Submit your logs of CartPole-v1, and a plot with environment steps on the x-axis and eval return on
    the y-axis
    """

    ## Load the data file
    HW3_DIR = Path(__file__).resolve().parents[1] # remonte au dossier parent
    EXP_DIR = HW3_DIR / "exp" / "CartPole-v1_dqn_sd1_20261109_174525"
    LOG_FILE = EXP_DIR / "log.csv"
    df = pd.read_csv(LOG_FILE)

    ## Select the correct column
    df_plot = df[["step", "Eval_AverageReturn"]]
    df_plot = df_plot.assign(Algorithme="DQN")

    ## Create the plot
    sns.set_theme()
    g = sns.relplot(data=df_plot,
                kind="line",
                x = "step",
                y="Eval_AverageReturn",
                hue="Algorithme",)

    g.set_xlabels("Environment step")
    g.set_ylabels("Eval return")
    g.set(title="DQN sur CartPole-v1")
    g.savefig(Path(__file__).resolve().parents[0] / "cartpole.png")

def main_25():
    """
    Plot the data visualisation of the section 2.5 of hw3 :

    Plot the eval return over training steps
    """

    ## Load the data file
    HW3_DIR = Path(__file__).resolve().parents[1] # remonte au dossier parent
    EXP_DIR = HW3_DIR / "exp" / "LunarLander-v2_dqn_sd1_20261109_175920"
    LOG_FILE = EXP_DIR / "log.csv"
    df = pd.read_csv(LOG_FILE)

    ## Select the correct column
    df_plot = df[["step", "Eval_AverageReturn"]]
    df_plot = df_plot.assign(Algorithme="DQN")

    ## Create the plot
    sns.set_theme()
    g = sns.relplot(data=df_plot,
                kind="line",
                x = "step",
                y="Eval_AverageReturn",
                hue="Algorithme",)

    g.set_xlabels("Environment step")
    g.set_ylabels("Eval return")
    g.set(title="DQN sur LunarLander")
    g.savefig(Path(__file__).resolve().parents[0] / "LunarLander.png")


def main_26():
    """
    Plot the data visualisation of the section 2.5 of hw3 :

    Plot the eval return over training steps
    """

    ## Load the data file
    HW3_DIR = Path(__file__).resolve().parents[1] # remonte au dossier parent
    EXP_DIR = HW3_DIR / "exp" / "LunarLander-v2_dqn_sd1_20261109_175920"
    LOG_FILE = EXP_DIR / "log.csv"
    df = pd.read_csv(LOG_FILE)

    ## Select the correct column
    df_plot = df[["step", "Eval_AverageReturn", "Train_EpisodeReturn"]]
    df_plot = df_plot.assign(Algorithme="DQN")

    ## Create the plot
    sns.set_theme()
    g = sns.relplot(data=df.set_index("step")[["Eval_AverageReturn", "Train_EpisodeReturn"]],
                    kind="line")

    g.set_xlabels("Environment step")
    g.set_ylabels("Eval return")
    g.set(title="DQN sur Pacman")
    g.savefig(Path(__file__).resolve().parents[0] / "Pacman.png")

if __name__ == "__main__":
    main_24()
    main_25()
    main_26()