import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import matplotlib.pyplot as plt

# 读取图像文件
image_path = 'Image transformation/img.jpg'  # 替换为你的图片路径
image = Image.open(image_path)
image_np = np.array(image)  # 将PIL图像转换为NumPy数组

# 定义增强序列
seq = iaa.Sequential([
    iaa.GaussianBlur(sigma=iap.Uniform(0.0, 1.0)),
    iaa.LinearContrast(alpha=iap.Choice([1.0, 1.5, 3.0], p=[0.5, 0.3, 0.2])),
    iaa.Affine(rotate=iap.Normal(0.0, 30), translate_px=iap.RandomSign(iap.Poisson(3))),
    iaa.AddElementwise(iap.Discretize((iap.Beta(0.5, 0.5) * 2 - 1.0) * 64)),
    iaa.Multiply(iap.Positive(iap.Normal(0.0, 0.1)) + 1.0)
])

# 应用增强序列到图像上
augmented_image = seq(images=np.array([image_np]))[0]

# 使用matplotlib显示图像
plt.imshow(augmented_image)
plt.axis('off')  # 不显示坐标轴
plt.title('Augmented Image')  # 设置图像标题
plt.show()