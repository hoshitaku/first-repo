import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt
ia.seed(1) # images = np.arry(16,64,64,3) and dtype uint8.
images = np.array(
    [ia.quokka( size = (64,64)) for _ in range(16)],
     dtype=np.uint8)
seq = iaa.Sequential([
     iaa.Fliplr(0.5),#horizontal flips
     iaa.Crop(percent=(0,0.1)),
     iaa.Sometimes(
         0.5,iaa.GaussianBlur(sigma=(0,0.5))
     ),
     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0,0.05*255), per_channel=0.5),
 ],random_order=True)
image_path = 'new_image.jpg'
images_aug = seq(images=images)
grid_image=ia.draw_grid(images_aug,4)
import imageio
imageio.imwrite("example_segmaps.jpg", grid_image)
plt.imshow(grid_image)
plt.axis('off')
plt.show()