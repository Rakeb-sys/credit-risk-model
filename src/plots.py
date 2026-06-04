import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_corr(df):
    # 2. Compute the Pearson correlation matrix
    correlation_matrix = df.corr()

    # 3. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # 4. Generate a mask for the upper triangle (optional, to avoid duplicate data)
    import numpy as np

    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # 5. Draw the heatmap
    sns.heatmap(
        correlation_matrix,  # Hides the redundant top half of the matrix
        annot=True,  # Writes the correlation numbers inside the squares
        fmt=".2f",  # Rounds the numbers to 2 decimal places
        cmap="coolwarm",  # Red = Positive, Blue = Negative correlation
        vmin=-1,
        vmax=1,  # Forces the color scale limits between -1 and 1
        square=True,  # Makes all cells perfectly square
        linewidths=0.5,  # Adds clean borders between cells
        ax=ax,
    )

    ax.set_title("Correlation Matrix of Numerical Features", fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()
