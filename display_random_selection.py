import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

path = './run_2024-02-08 curated/shard_0'

imgs = glob.glob(os.path.join(path, '*.jpg'))

imgs = np.random.choice(imgs, 100)

fig, axes = plt.subplots(10, 10, figsize=(20, 20))
for i, ax in enumerate(axes.flatten()):
    img = Image.open(imgs[i]).resize((224, 224))
    ax.imshow(np.array(img))
    ax.axis('off')
plt.tight_layout()
plt.savefig('random_curated_images.png', dpi=300)