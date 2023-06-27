# transfer files from one all to another all
# split files from all into train, valid, test
# specify percent split

import shutil
import argparse
import glob
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--profile', help='name of profile')
    parser.add_argument('-d1', '--dataset1', help='name of origin data directory')
    parser.add_argument('-d2', '--dataset2', help='name of destination data directory')
    parser.add_argument('-f', '--function', help='choose between:'
                                                 '"A2A": from dataset1/all to dataset2/all,'
                                                 '"A2TV": from dataset1/all to (dataset1/train, dataset1/valid)'
                                                 '"A2TVT": from dataset1/all '
                                                 'to (dataset1/train, dataset1/valid, dataset1/test)')
    args = vars(parser.parse_args())

    return args


def get_input():
    args = get_args()
    profile, dataset1, dataset2, function = \
        args.get('profile'), args.get('dataset1'), args.get('dataset2'), args.get('function')
    return profile, dataset1, dataset2, function


def a2a(origin_path, destination_path):
    file_list = os.listdir(origin_path)
    for file_name in file_list:
        source_file = os.path.join(origin_path, file_name)
        destination_file = os.path.join(destination_path, file_name)
        shutil.copy(source_file, destination_file)



def a2tv(origin_path, num_files):

    split_path = origin_path.replace("/all", "")
    train_percent = check("A2TV", float(input("specify percentage allocated to training set (0.X): ")), None)

    train_num = int(num_files*float(train_percent))
    valid_num = num_files - train_num

    png_files = glob.glob(f"{origin_path}/*.png")
    png_files_iter = iter(png_files)

    copy_file(train_num, png_files_iter, split_path, "train")
    copy_file(valid_num, png_files_iter, split_path, "valid")


def a2tvt(origin_path, num_files):

    split_path = origin_path.replace("/all", "")
    train_percent = float(input("specify percentage allocated to training set (0.X): "))
    valid_percent = float(input("specify percentage allocated to training set (0.X): "))
    train_percent = check("A2TVT", train_percent, valid_percent)

    train_num = int(num_files*train_percent)
    valid_num = int(num_files*valid_percent)
    test_num = num_files - train_num - valid_num

    png_files = glob.glob(f"{origin_path}/*.png")
    png_files_iter = iter(png_files)

    copy_file(train_num, png_files_iter, split_path, "train")
    copy_file(valid_num, png_files_iter, split_path, "valid")
    copy_file(test_num, png_files_iter, split_path, "test")


def copy_file(num_files, png_files_iter, split_path, folder):
    for i in range(num_files):
        try:
            img_path = next(png_files_iter)
            xml_path = os.path.splitext(img_path)[0] + ".xml"
            shutil.copy(img_path, f"{split_path}/{folder}")
            if os.path.exists(xml_path):
                shutil.copy(xml_path, f"{split_path}/{folder}")
        except StopIteration:
            return


def check(transfer_type, x, y):
    if transfer_type == "A2TV":
        if not 0 < x <= 1:
            return 1.0
        else:
            return x
    if transfer_type == "A2TVT":
        if (x + y > 1) or x < 0 or y < 0:
            return 1.0
        else:
            return x


if __name__ == '__main__':
    profile, dataset1, dataset2, function = get_input()
    origin_path = f"../../profiles/{profile}/data/{dataset1}/all"

    num_files = len(glob.glob(f"{origin_path}/*.png"))

    if function == "A2A":
        destination_path = f"../../profiles/{profile}/data/{dataset2}/all"
        a2a(origin_path, destination_path)

    if function == "A2TV":
        print(f"\ncurrent number of images: {num_files}")
        a2tv(origin_path, num_files)

    if function == "A2TVT":
        print(f"\ncurrent number of images: {num_files}")
        a2tvt(origin_path, num_files)

