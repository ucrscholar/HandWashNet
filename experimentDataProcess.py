import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from PIL import Image

masks = glob.glob("C:/Users/sheng/Videos/experimentsData/ilab_Gesture A_2019-11-28_15-14-15_top_Color_*.png")
orgs = list(map(lambda x: x.replace("Color", "Color"), masks))
# masks = glob.glob("C:/Users/sheng/Downloads/HGR1/*.bmp")
# orgs = list(map(lambda x: x.replace("bmp", "jpg"), masks))

# %%

imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    imgs_list.append(np.array(Image.open(image).resize((384, 384))))
    masks_list.append(np.array(Image.open(mask).resize((384, 384))))

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

# %%

print(imgs_np.shape, masks_np.shape)

print(imgs_np.max(), masks_np.max())

# %%

x = np.asarray(imgs_np, dtype=np.float32) / 255
y = np.asarray(masks_np, dtype=np.float32) / 255

f, ax = plt.subplots(10, 12, sharex=True, sharey=True, figsize=(5 * 1, 5 * 1))

ax[0, 0].set_title("Image A", fontsize=15)
ax[0, 0].set_axis_off()
ax[0, 1].set_title("Image B", fontsize=15)
ax[0, 1].set_axis_off()
for row in range(0, 10):
    for col in range(0, 12):
        ax[row, col].imshow(x[(row+0)*12 + col], cmap='jet')
        ax[row, col].set_title(str((row+0)*12 + col), fontsize=10)  # increase or decrease y as needed
        ax[row, col].set_axis_off()
    print('row: ' + str(row))
plt.savefig('ax.png', format="png")
plt.show()
