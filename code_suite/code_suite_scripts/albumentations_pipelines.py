import albumentations as A
import cv2 as cv

transform_pipeline_no_manual = A.Compose(
    [
        A.SomeOf([
            A.OneOf([A.HorizontalFlip(), A.VerticalFlip()]),
            A.OneOf([A.RandomRotate90(), A.Transpose(), A.ShiftScaleRotate()]),
            A.OneOf([A.ElasticTransform(), A.GridDistortion()]),
            A.RandomSizedCrop(min_max_height=(500, 600), height=1000, width=700, p=1),
            A.OneOf([A.ChannelShuffle(), A.RandomBrightnessContrast(), A.CLAHE(), A.ToGray()]),
            A.OneOf([A.Solarize(), A.RandomSunFlare(), A.Spatter()]),
            A.OneOf([A.GlassBlur(sigma=0.5, max_delta=2), A.Downscale(interpolation=cv.INTER_NEAREST),
                     A.MotionBlur(blur_limit=15)]),
            ], n=3)
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

transform_pipeline_manual = A.Compose(
    [
        A.SomeOf([
            A.OneOf([A.HorizontalFlip(), A.VerticalFlip()]),
            A.OneOf([A.RandomRotate90(), A.Transpose(), A.ShiftScaleRotate()]),
            A.OneOf([A.ElasticTransform(), A.GridDistortion()]),
            A.RandomSizedCrop(min_max_height=(500, 600), height=1000, width=700, p=1),
            A.OneOf([A.ChannelShuffle(), A.RandomBrightnessContrast(), A.CLAHE(), A.ToGray()]),
            A.OneOf([A.Solarize(), A.RandomSunFlare(), A.Spatter()]),
            A.OneOf([A.GlassBlur(sigma=0.5, max_delta=2), A.Downscale(interpolation=cv.INTER_NEAREST),
                     A.MotionBlur(blur_limit=15)]),
            ], n=2),
        A.OneOf([A.GridDropout(random_offset=True), A.RandomGridShuffle(),
                 A.CoarseDropout(max_holes=25, max_height=25, max_width=25)])
    ])
