import imgaug.augmenters as iaa
aug = iaa.OneOf([
    iaa.Affine(rotate=45),
    iaa.AdditiveGaussianNoise(scale=0.2*255),
    iaa.Add(50,per_channel=True),
    iaa.Sharpen(alpha=0.5)
])