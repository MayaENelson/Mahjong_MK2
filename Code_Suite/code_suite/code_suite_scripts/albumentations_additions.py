# get image set to 1000 images
# take available image and pascal VOC label
# cycle through each photo until 1000 is reached
# edit the image and pascal VOC label
# save to folder

import albumentations as A
import cv2
import cv2 as cv
import os
import glob
import argparse
import numpy as np
import itertools
import random
import string
from xml.etree import ElementTree as et

transform_pipeline = A.Compose(
    [
        A.SomeOf([
            A.OneOf([A.HorizontalFlip(), A.VerticalFlip()]),
            A.OneOf([A.RandomRotate90(), A.Transpose(), A.ShiftScaleRotate()]),
            A.OneOf([A.ElasticTransform(), A.GridDistortion()]),
            A.RandomSizedCrop(min_max_height=(500, 600), height=1000, width=700, p=1),
            A.OneOf([A.ChannelShuffle(), A.RandomBrightnessContrast(), A.CLAHE(), A.ToGray()]),
            A.OneOf([A.Solarize(), A.RandomSunFlare(), A.Spatter()]),
            A.OneOf([A.GlassBlur(sigma=0.5, max_delta=2), A.Downscale(interpolation=cv2.INTER_NEAREST),
                     A.MotionBlur(blur_limit=15)]),
            ], n=3)
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


class OriginalImageset:
    def __init__(self, root_path, txt_path):
        with open(txt_path, "r") as file:
            image_dirs = eval(file.read())

        self.__root_path = root_path
        self.__image_dirs = image_dirs
        self.__images = []

    def convert_numpy(self):
        for image_dir in self.__image_dirs:
            img = cv.cvtColor(cv.imread(image_dir), cv.COLOR_BGR2RGB)
            self.__images.append(img)

    def view_images(self):
        for img in self.__images:
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            cv.imshow('Original Images', img)
            cv.waitKey(0)

    def get_images(self):
        return self.__images

    def get_image_dirs(self):
        return self.__image_dirs


class NewImageset:
    def __init__(self, root_path):
        self.__root_path = root_path
        self.__image_dirs = []
        self.__images = []

    def create_images(self, orig_images, orig_image_dirs, transform_pipeline, number):
        number_files = len(glob.glob(f"{self.__root_path}/*.png"))
        # re-use all the available images to make a set of 'number' available images to alter
        orig_images = itertools.islice(itertools.cycle(orig_images), number - number_files)
        orig_image_dirs = itertools.islice(itertools.cycle(orig_image_dirs), number - number_files)

        for img, img_dir in zip(orig_images, orig_image_dirs):

            boxes, labels = convert_xml(img_dir)

            transformed_dict = transform_pipeline(image=img, bboxes=boxes, labels=labels)
            transformed_img = transformed_dict['image']
            transformed_bbox = transformed_dict['bboxes']
            transformed_labels = transformed_dict['labels']

            transformed_img = cv.cvtColor(transformed_img, cv.COLOR_RGB2BGR)
            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            cv.imwrite(f"{self.__root_path}/{random_string}.png", transformed_img)

            print(boxes, labels)

    def view_images(self):
        for img in self.__images:
            cv.imshow('Original Images', img)
            cv.waitKey(0)


def initialize_original(root_path):

    image_dirs = glob.glob(f"{root_path}/*.png")
    rename_original(image_dirs)
    image_dirs = glob.glob(f"{root_path}/*.png")

    txt_path = f"{root_path}/orig_dirs.txt"

    with open(txt_path, "w") as file:
        file.write(repr(image_dirs))


def rename_original(image_dirs):
    for idx, img_dir in enumerate(image_dirs):
        xml_path = os.path.splitext(img_dir)[0] + ".xml"

        new_img_path = os.path.join(os.path.dirname(img_dir), f"original_{idx}.png")
        new_xml_path = os.path.join(os.path.dirname(xml_path), f"original_{idx}.xml")
        os.rename(img_dir, new_img_path)
        os.rename(xml_path, new_xml_path)



def convert_xml(img_dir):
    boxes = []
    labels = []

    xml_path = os.path.splitext(img_dir)[0] + ".xml"

    tree = et.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        xmin = int(float(obj.find('bndbox').find('xmin').text))
        ymin = int(float(obj.find('bndbox').find('ymin').text))
        xmax = int(float(obj.find('bndbox').find('xmax').text))
        ymax = int(float(obj.find('bndbox').find('ymax').text))

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(obj.find('name').text)

    return boxes, labels


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--profile', help='name of profile')
    parser.add_argument('-d', '--dataset', help='name of dataset')
    parser.add_argument('-n', '--number', help='number of images to create from existing')
    parser.add_argument('-f', '--first', action='store_true', help='flag for first launch')
    args = vars(parser.parse_args())

    return args


def transforms():
    pass


if __name__ == "__main__":
    args = get_args()
    profile = args['profile']
    dataset = args['dataset']
    number = int(args['number'])


    # deal with file not found
    root_path = f"../../profiles/{profile}/data/{dataset}/all"

    if args['first']:
        initialize_original(root_path)
    if os.path.isfile(f"{root_path}/orig_dirs.txt"):
        orig_image_set = OriginalImageset(root_path, f"{root_path}/orig_dirs.txt")
        orig_image_set.convert_numpy()
        var_orig_image_set = orig_image_set.get_images()
        orig_image_dirs = orig_image_set.get_image_dirs()
        new_image_set = NewImageset(root_path)
        new_image_set.create_images(var_orig_image_set, orig_image_dirs, transform_pipeline, number)
    else:
        print("use the first flag '-f' to make 'orig_dirs.txt'")
