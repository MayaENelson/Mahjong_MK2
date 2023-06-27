import albumentations as A
import cv2 as cv
import random

pipeline_no_manual = A.Compose(
    [
        A.RandomSizedCrop(min_max_height=(300, 500),
                          height=random.randint(100, 800),
                          width=random.randint(100, 800), p=0.5),
        A.SomeOf([
            A.OneOf([A.HorizontalFlip(), A.VerticalFlip()]),
            A.OneOf([A.RandomRotate90(), A.Transpose(), A.ShiftScaleRotate()]),
            A.OneOf([A.ElasticTransform(), A.GridDistortion()]),
            A.OneOf([A.ChannelShuffle(), A.RandomBrightnessContrast(), A.CLAHE(), A.ToGray()]),
            A.OneOf([A.Solarize(), A.RandomSunFlare(), A.Spatter()]),
            A.OneOf([A.GlassBlur(sigma=0.5, max_delta=2), A.Downscale(interpolation=cv.INTER_NEAREST),
                     A.MotionBlur(blur_limit=15)]),
            ], n=3)
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

pipeline_manual = A.Compose(
    [
        A.RandomSizedCrop(min_max_height=(300, 500),
                          height=random.randint(100, 800),
                          width=random.randint(100, 800), p=0.5),
        A.SomeOf([
            A.OneOf([A.HorizontalFlip(), A.VerticalFlip()]),
            A.OneOf([A.RandomRotate90(), A.Transpose(), A.ShiftScaleRotate()]),
            A.OneOf([A.ElasticTransform(), A.GridDistortion()]),
            A.OneOf([A.ChannelShuffle(), A.RandomBrightnessContrast(), A.CLAHE(), A.ToGray()]),
            A.OneOf([A.Solarize(), A.RandomSunFlare(), A.Spatter()]),
            A.OneOf([A.GlassBlur(sigma=0.5, max_delta=2), A.Downscale(interpolation=cv.INTER_NEAREST),
                     A.MotionBlur(blur_limit=15)]),
            ], n=2),
        A.OneOf([A.GridDropout(random_offset=True), A.RandomGridShuffle(),
                 A.CoarseDropout(max_holes=25, max_height=25, max_width=25)])
    ])

