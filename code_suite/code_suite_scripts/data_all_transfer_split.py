# transfer files from one all to another all
# split files from all into train, valid, test
# specify percent split

import shutil
import argparse


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
        args.get('profile'), args.get('dataset1'), args.get('dataset1'), args.get('function')
    return profile, dataset1, dataset2, function


def a2a(origin_path, destination_path):
    shutil.copy(origin_path, destination_path)


def a2tv(origin_path):
    pass


def a2tvt(origin_path):
    pass


if __name__ == '__main__':
    profile, dataset1, dataset2, function = get_input()
    origin_path = f"../../profiles/{profile}/data/{dataset1}/all"

    if function is "A2A" and dataset1 is not None and dataset2 is not None:
        destination_path = f"../../profiles/{profile}/data/{dataset2}/all"
        a2a(origin_path, destination_path)
    else:
        print("specify both datasets using '-d1' and '-d2' flags")

    if function is "A2TV":
        pass

    if function is "A2TVT":
        pass

