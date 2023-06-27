# creates copies of and augment the existing dataset to increase dataset size
# cli args: -p
#           -d
#           -n
#           -pp
#           -m
#           -f
#           -s

import cv2 as cv
import os
import glob
import argparse
import itertools
import random
import string
from xml.etree import ElementTree as et
import importlib
import albumentations as A
from universal_fx import extract_bbox_xml


class PascalVOC:
    def __init__(self, filename, width, height):
        self.root = et.Element("annotation")
        self.folder = et.SubElement(self.root, "folder")
        self.filename = et.SubElement(self.root, "filename")
        self.filename.text = filename
        self.size = et.SubElement(self.root, "size")
        self.width = et.SubElement(self.size, "width")
        self.width.text = str(width)
        self.height = et.SubElement(self.size, "height")
        self.height.text = str(height)

    def add_bbox(self, label, xmin, ymin, xmax, ymax):
        obj = et.SubElement(self.root, "object")
        name = et.SubElement(obj, "name")
        name.text = label
        bndbox = et.SubElement(obj, "bndbox")
        xmin_elem = et.SubElement(bndbox, "xmin")
        xmin_elem.text = str(xmin)
        ymin_elem = et.SubElement(bndbox, "ymin")
        ymin_elem.text = str(ymin)
        xmax_elem = et.SubElement(bndbox, "xmax")
        xmax_elem.text = str(xmax)
        ymax_elem = et.SubElement(bndbox, "ymax")
        ymax_elem.text = str(ymax)

    def write_xml(self, filepath):
        tree = et.ElementTree(self.root)
        tree.write(filepath)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--profile', help='name of profile')
    parser.add_argument('-d', '--dataset', help='name of dataset directory')
    parser.add_argument('-n', '--number', help='number of images to create using transformation pipeline')
    parser.add_argument('-pp', '--pipeline', help='name of transformation pipeline')
    parser.add_argument('-m', '--manual', action='store_true', help='flag for pipeline with no bbox function')
    parser.add_argument('-f', '--first', action='store_true', help='flag for first launch')
    parser.add_argument('-s', '--show', action='store_true', help='show number of images currently in dataset')
    args = vars(parser.parse_args())

    return args


def read_write_img(root_path, orig_image_dirs, number, transform_pipeline, transform_type):
    orig_image_dirs = itertools.islice(itertools.cycle(orig_image_dirs), number)

    for img_dir in orig_image_dirs:
        img = cv.cvtColor(cv.imread(img_dir), cv.COLOR_BGR2RGB)
        transformed_img_dict = make_transformed_dict(transform_type, transform_pipeline, img_dir, img, root_path)
        filename = write_image(transformed_img_dict)
        bbox_write_xml(transform_type, transformed_img_dict, filename, root_path)


def make_transformed_dict(transform_type, transform_pipeline, img_dir, img, root_path):
    transform_pipeline = set_pipeline(transform_pipeline)

    if transform_type == "non-manual":
        boxes, labels = extract_bbox_xml(img_dir)
        transformed_dict = transform_pipeline(image=img, bboxes=boxes, labels=labels)
    else:
        transformed_dict = transform_pipeline(image=img)

    return transformed_dict


def set_pipeline(transform_pipeline):
    transform_pipeline = instance_request(transform_pipeline)
    max_height, _ = get_max_hw(root_path)
    set_rand_crop(transform_pipeline, max_height)

    return transform_pipeline


def set_rand_crop(transform_pipeline, max_height):
    for transform in transform_pipeline.transforms:
        if isinstance(transform, A.RandomSizedCrop):
            transform.min_max_height = (300, max_height)
            transform.height = random.randint(100, 800)
            transform.width = random.randint(100, 800)
            break


def write_image(transformed_dict):
    transformed_img, _, _ = get_dict_vals(transformed_dict)
    transformed_img = cv.cvtColor(transformed_img, cv.COLOR_RGB2BGR)

    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    filename = f"{random_string}.png"

    cv.imwrite(f"{root_path}/{filename}", transformed_img)

    return filename


def get_dict_vals(transformed_dict):
    transformed_img = transformed_dict['image']
    transformed_bbox = transformed_dict.get('bboxes')
    transformed_labels = transformed_dict.get('labels')

    return transformed_img, transformed_bbox, transformed_labels


def bbox_write_xml(transform_type, transformed_dict, filename, root_path):
    transformed_img, transformed_bbox, transformed_labels = get_dict_vals(transformed_dict)

    if transformed_bbox:
        transformed_bbox = [[round(num, 1) for num in sublist] for sublist in transformed_bbox]

    if transform_type == "non-manual":
        height, width, _ = transformed_img.shape
        xml_writer = PascalVOC(filename, height, width)

        for label, bbox in zip(transformed_labels, transformed_bbox):
            xml_writer.add_bbox(label, bbox[0], bbox[1], bbox[2], bbox[3])
        xml_writer.write_xml(f"{root_path}/{filename.split('.')[0]}.xml")


def initialize_original(root_path):
    image_dirs = glob.glob(f"{root_path}/*.png")
    rename_overwrite(image_dirs)
    image_dirs = glob.glob(f"{root_path}/*.png")

    txt_path = f"{root_path}/orig_dirs.txt"

    with open(txt_path, "w") as txt_file:
        txt_file.write(repr(image_dirs))


def rename_overwrite(image_dirs):
    for idx, img_dir in enumerate(image_dirs):
        xml_path = os.path.splitext(img_dir)[0] + ".xml"

        new_xml_path = rename_original_files(idx, img_dir, xml_path)
        overwrite_original_xmls(idx, new_xml_path)


def rename_original_files(idx, img_dir, xml_path):
    new_img_path = os.path.join(os.path.dirname(img_dir), f"original_{idx}.png")
    new_xml_path = os.path.join(os.path.dirname(xml_path), f"original_{idx}.xml")
    os.rename(img_dir, new_img_path)
    os.rename(xml_path, new_xml_path)

    return new_xml_path


def overwrite_original_xmls(idx, new_xml_path):
    tree = et.parse(new_xml_path)
    root = tree.getroot()
    title = root.find("filename")
    title.text = f"original_{idx}.png"
    tree.write(new_xml_path)


def instance_request(instance_name):
    module = importlib.import_module("albumentations_pipelines")
    instance = getattr(module, instance_name)
    return instance


def convert_number_img(number_img):
    if number_img is None:
        number_img = 0
    else:
        number_img = int(number_img)
    return number_img


def get_input():
    args = get_args()
    profile, dataset, number, pipeline, manual, first, show, = \
        args.get('profile'), args.get('dataset'), args.get('number'), args.get('pipeline'), \
        args['manual'], args['first'], args['show']
    number = convert_number_img(number)
    return profile, dataset, number, pipeline, manual, first, show


def get_pipeline_check(manual, transform_pipeline):
    if manual:
        transform_type = "manual"
    else:
        transform_type = "non-manual"

    transform_pipeline_check = instance_request(transform_pipeline)
    return transform_type, transform_pipeline_check


def first_show_flags_fx(first, show, root_path):
    if first and not os.path.isfile(f"{root_path}/orig_dirs.txt"):
        initialize_original(root_path)
    if show:
        number_files = len(glob.glob(f"{root_path}/*.png"))
        print(f"\ncurrent number of images: {number_files}")


def get_orig_img_dirs(root_path):
    with open(f"{root_path}/orig_dirs.txt", "r") as file:
        orig_image_dirs = eval(file.read())
    return orig_image_dirs


def get_max_hw(root_path):
    smallest_size = float('inf')
    smallest_file = None
    orig_image_dirs = get_orig_img_dirs(root_path)

    for img_dir in orig_image_dirs:
        image_size = os.path.getsize(img_dir)

        if image_size < smallest_size:
            smallest_size = image_size
            smallest_file = img_dir

    img = cv.imread(smallest_file)
    max_height, max_width, _ = img.shape

    return max_height, max_width


if __name__ == "__main__":
    profile, dataset, number, pipeline, manual, first, show = get_input()
    root_path = f"../../profiles/{profile}/data/{dataset}/all"
    exists = os.path.exists(root_path)

    if exists:
        first_show_flags_fx(first, show, root_path)
        transform_type, transform_pipeline_check = get_pipeline_check(manual, pipeline)

        if os.path.isfile(f"{root_path}/orig_dirs.txt") and isinstance(transform_pipeline_check, A.Compose):
            orig_image_dirs = get_orig_img_dirs(root_path)
            read_write_img(root_path, orig_image_dirs, number, pipeline, transform_type)
        else:
            print("check: specify a pipeline")
            print("check: use the first flag '-f' to make 'orig_dirs.txt'")
    else:
        print("check: filepath")
