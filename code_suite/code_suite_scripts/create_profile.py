# make file directory for new profile and update profile
# cli args: -p
#           -d
#           -m
#           -u
#           -c

import os
import argparse

delimiter = ","
new_profile = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--profile', help='name of profile')
    parser.add_argument('-d', '--dataset', help='name of dataset directories, separate with ","')
    parser.add_argument('-m', '--models', help='name of model directories, separate with ","')
    parser.add_argument('-u', '--update', action='store_true', help='update only; do not attempt to create profile')
    parser.add_argument('-c', '--copy', action='store_true', help='make respective model folders for dataset folders')
    args = vars(parser.parse_args())

    return args


def make_profile(profile):
    profile_path = f"../../profiles/{profile}"
    new_folders = [profile_path, f"{profile_path}/data", f"{profile_path}/models", f"{profile_path}/models/backbones",
                   f"{profile_path}/models/un-trained_models", f"{profile_path}/util"]

    if not os.path.exists(profile_path):
        for folder in new_folders:
            os.makedirs(folder)
        print(f"\n'{profile}' created")


def make_data_dir(profile, data_dirs):
    for ddir in data_dirs:
        data_path = f"../../profiles/{profile}/data/{ddir}"
        new_folders = [f"{data_path}/train", f"{data_path}/train/vis", f"{data_path}/valid", f"{data_path}/valid/vis",
                       f"{data_path}/test", f"{data_path}/test/vis", f"{data_path}/all"]

        if not os.path.exists(data_path):
            for folder in new_folders:
                os.makedirs(folder)
            print(f"\n'{ddir}' dataset in '{profile}' created")
        else:
            print(f"\nERROR: dataset '{ddir}' already exists")


def make_model_dir(profile, model_dirs):
    for mdir in model_dirs:
        model_path = f"../../profiles/{profile}/models/{mdir}"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            print(f"\n'{mdir}' model in '{profile}' created")
        else:
            print(f"\nERROR: model '{mdir}' already exists")


def get_input():
    args = get_args()
    profile, data, models, update, copy = \
        args.get('profile'), args.get('dataset'), args.get('models'), args['update'], args['copy']
    return profile, data, models, update, copy


if __name__ == "__main__":
    profile, data, models, update, copy = get_input()

    if not os.path.exists(f"../../profiles/{profile}"):
        make_profile(profile)
        new_profile = True
    if update or new_profile:
        if data is not None:
            data_dirs = data.split(delimiter)
            make_data_dir(profile, data_dirs)
        if models is not None:
            model_dirs = models.split(delimiter)
            make_model_dir(profile, model_dirs)
        elif copy:
            model_dirs = data.split(delimiter)
            make_model_dir(profile, model_dirs)
    else:
        print("\nno actions taken")
