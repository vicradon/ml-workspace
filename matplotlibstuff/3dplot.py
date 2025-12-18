import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

Z_map = {
    "Z = X² + Y²":            X**2 + Y**2,
    "Z = -(X² + Y²)":         -(X**2 + Y**2),
    "Z = X² - Y²":            X**2 - Y**2,
    "Z = -(X² - Y²)":         -(X**2 - Y**2),
    "Z = X² · Y²":            X**2 * Y**2,
    "Z = -(X² · Y²)":         -(X**2 * Y**2),
    "Z = X² / Y²":            np.where(Y != 0, X**2 / Y**2, np.nan),
    "Z = -(X² / Y²)":         np.where(Y != 0, -(X**2 / Y**2), np.nan),
}

fig = plt.figure(figsize=(20, 10))

for i, (title, Z) in enumerate(Z_map.items(), start=1):
    ax = fig.add_subplot(2, 4, i, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="winter")
    ax.text2D(
        0.5, -0.12, title,
        transform=ax.transAxes,
        ha="center",
        va="top"
    )
    print("Plotted", title)

import pathlib

plots_dir = pathlib.Path("plots")
plots_dir.mkdir(exist_ok=True)

print("Now creating image from plots...")
plt.savefig(plots_dir / "3d-plots.png", dpi=300, bbox_inches="tight")