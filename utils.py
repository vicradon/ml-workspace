import os
import math
import glob
import subprocess
import re
from PIL import Image
from matplotlib import pyplot as plt

def plot_state(label, iteration, x_val, y_val, x_range, y_range):
    skip = max(1, int(math.log10(iteration + 1) * 10))

    if iteration % skip != 0:
        return

    plt.figure(figsize=(6, 4))
    plt.plot(x_range, y_range, label=label)
    plt.scatter([x_val], [y_val], s=50)

    plt.title(f"Iteration {iteration} — x={x_val:.5f}, f(x)={y_val:.5f}")
    plt.xlabel("x")
    plt.ylabel("f(x)")

    output_file = f"plots/plot_iter_{iteration}.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()



def create_gif_from_plots(remove_files=False):
    # Get all PNG files and sort them numerically
    pattern = r'plot_iter_(\d+)\.png'
    files = glob.glob('plots/*.png')

    # Extract numbers and sort
    files_with_numbers = []
    for file in files:
        match = re.search(pattern, file)
        if match:
            number = int(match.group(1))
            files_with_numbers.append((number, file))

    # Sort by numerical order
    files_with_numbers.sort()
    sorted_files = [file for _, file in files_with_numbers]

    # Open images
    images = []
    for file_path in sorted_files:
        img = Image.open(file_path)
        images.append(img)

    # Save as GIF
    images[0].save(
        'plots/animation.gif',
        save_all=True,
        append_images=images[1:],
        duration=100,  # milliseconds between frames (10 = 0.01s, but PIL uses 100ms min)
        loop=1
    )

    if remove_files:
        for fp in sorted_files:
            os.remove(fp)

    print(f"Created GIF with {len(images)} frames")

