import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.linear_model import Perceptron
import urllib.request
import os

# 1. SETUP
os.makedirs('plots', exist_ok=True)
os.makedirs('assets', exist_ok=True)

# Function to fetch the PNGs
def get_emoji_img(name, url):
    path = f"assets/{name}.png"
    if not os.path.exists(path):
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(path, 'wb') as out_file:
            out_file.write(response.read())
    return plt.imread(path)

# Direct PNG links for your specific requested variants
# Nails (Medium-Dark Skin Tone) & Moai
nails_url = "https://openmoji.org/php/download_asset.php?type=emoji&emoji_hexcode=1F485&emoji_variant=color"
moai_url = "https://openmoji.org/data/color/png/64/1F5FF.png"

nails_img = get_emoji_img("nails", nails_url)
moai_img = get_emoji_img("moai", moai_url)

# 2. DATA (15 Training Boys)
X_train = np.array([
    [0.4, 0.8], [0.6, 0.2], [0.0, 0.8], [0.6, 0.6], [0.4, 0.4], 
    [1.0, 0.0], [0.6, 0.4], [0.2, 0.6], [0.6, 0.4], [0.2, 1.0], 
    [0.6, 0.4], [0.4, 0.6], [0.0, 0.4], [0.4, 0.6], [0.4, 0.2]
])
y_train = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1])

# 3. TRAIN
clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train, y_train)

# 4. PLOT WITH IMAGE MARKERS
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# Decision Background
h = .005
xx, yy = np.meshgrid(np.arange(-0.1, 1.1, h), np.arange(-0.1, 1.1, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.1, cmap=plt.cm.RdYlBu)

# Custom function to place images on the grid
def imscatter(x, y, image, ax, zoom=0.5):
    for x0, y0 in zip(x, y):
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        ax.add_artist(ab)

imscatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], moai_img, ax=ax)
imscatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], nails_img, ax=ax)

# 5. BOUNDARY LINE
w1, w2 = clf.coef_[0]
b = clf.intercept_[0]
x_line = np.linspace(0, 1, 100)
y_line = -(w1 / w2) * x_line - (b / w2)
ax.plot(x_line, y_line, color='green', linestyle='--', linewidth=2.5, label='The Bro-Boundary')

# 6. AESTHETICS
ax.set_title('The Perceptron Bro Audit: 🗿 vs 💅', fontsize=16, fontweight='bold')
ax.set_xlabel('Zestiness Score ($x_1$)', fontsize=12)
ax.set_ylabel('Stoicism Score ($x_2$)', fontsize=12)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, linestyle=':', alpha=0.3)
ax.legend(loc='upper right')

plt.savefig('plots/perceptron-is-bro-gay.png', dpi=300, bbox_inches='tight')
plt.show()