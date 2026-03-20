import matplotlib.pyplot as plt
import numpy as np

# 1. Setup the data points
x_not_gay = [0.1, 0.2, 0.4, 0.5]
y_not_gay = [0.2, 0.1, 0.4, 0.3]

x_gay = [0.6, 0.7, 0.8, 0.9]
y_gay = [0.7, 0.5, 0.9, 0.2]

# 2. Define the decision boundary
x_boundary = np.linspace(0, 1, 500) # Increased sampling for smoother lines
y_boundary = x_boundary

# 3. Create the plot with high resolution
# 'dpi=300' ensures the output is sharp
plt.figure(figsize=(10, 8), dpi=300)

# Fill background regions
plt.fill_between(x_boundary, y_boundary, 1, color='blue', alpha=0.1, label='Output: 1 (Gay)')
plt.fill_between(x_boundary, 0, y_boundary, color='red', alpha=0.1, label='Output: 0 (Not Gay)')

# Plot data points
plt.scatter(x_not_gay, y_not_gay, marker='x', color='red', s=120, linewidth=2, label='Is Not Gay (0)')
plt.scatter(x_gay, y_gay, marker='o', color='blue', s=120, edgecolors='white', label='Is Gay (1)')

# Plot the decision boundary line
plt.plot(x_boundary, y_boundary, color='green', linestyle='--', linewidth=2.5, label='Perceptron Decision Boundary')

# 4. Aesthetics and Labels
plt.title('Perceptron: "Is Gay?" Classifier Plot', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Zestiness Score ($x_1$)', fontsize=12)
plt.ylabel('Stoicism Score ($x_2$)', fontsize=12)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', frameon=True, shadow=True)

# 5. Save the high-res file
# This will save a crisp 3000x2400 pixel image in your directory
plt.savefig('perceptron_plot_high_res.png', dpi=300, bbox_inches='tight')

plt.show()
