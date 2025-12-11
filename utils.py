import os
import glob
import subprocess
from matplotlib import pyplot as plt

def make_gif_from_plots(dir):

    # Collect PNG files in {dir} (sorted for deterministic frame order)
    png_files = sorted(glob.glob(os.path.join(dir, "*.png")))

    # Build the ImageMagick command
    cmd = [
        "convert",
        "-delay", "20",   # 20 = 0.2 seconds per frame
        "-loop", "0"      # infinite loop
    ] + png_files + [f"{dir}/output.gif"]

    subprocess.run(cmd, check=True)

    subprocess.run(["rm", f"{dir}/*.png"])


def plot_state(label, iteration, skip_after, x_val, y_val, x_range, y_range):
    if iteration%skip_after != 0: return

    plt.figure(figsize=(6, 4))

    plt.plot(x_range, y_range, label=label)
    plt.scatter([x_val], [y_val], s=50)

    plt.title(f"Iteration {iteration} — x={x_val:.5f}, f(x)={y_val:.5f}")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    output_file = f"plots/plot_iter_{iteration}.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")