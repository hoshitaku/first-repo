import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image
import imageio
from matplotlib import pyplot as plt
# 同样的数据增强操作，各自参数不一样
# 一个包含16张图片的文件列表  
image_files = ['img.jpg', 'img.jpg', 'img.jpg', 'img.jpg', 'img.jpg', 'img.jpg', 'img.jpg', 'img.jpg',
               'img.jpg', 'img.jpg', 'img.jpg', 'img.jpg', 'img.jpg', 'img.jpg', 'img.jpg', 'img.jpg']

# 读取图片并调整大小到64x64  
images = np.array([
    np.array(Image.open(file).resize((64, 64)))
    for file in image_files
], dtype=np.uint8)

# 确保图片的数量和形状正确  
assert images.shape == (16, 64, 64, 3)

# 设置随机种子，使得每次增强效果一致  
ia.seed(1)

# 定义增强序列  
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # 水平翻转  
    iaa.Crop(percent=(0, 0.1)),
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
], random_order=True)

# 对图片进行增强  
images_aug = seq(images=images)

# 绘制网格图片  
grid_image = ia.draw_grid(images_aug, cols=4)  # 使用cols参数来指定每行显示的图片数  

# 保存网格图片  
imageio.imwrite("example_segmaps.jpg", grid_image)

# 显示网格图片  
plt.imshow(grid_image)
plt.axis('off')
plt.show()