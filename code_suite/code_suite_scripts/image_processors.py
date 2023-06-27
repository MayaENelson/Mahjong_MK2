import cv2 as cv
import argparse
import glob

### ADAPTIVE MEAN THRESHOLD
# img = cv.imread('sudoku.png', cv.IMREAD_GRAYSCALE)
# img = cv.medianBlur(img,5)
# th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#  cv.THRESH_BINARY,11,2)

### CANNY
# img = cv.imread('sudoku.png', cv.IMREAD_GRAYSCALE)
# edges = cv.Canny(img,100,200)
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# look at my other implementation?

### CLAHE
# img = cv.imread('tsukuba_l.png', cv.IMREAD_GRAYSCALE)
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)

### OTSU
# img = cv.imread('noisy2.png', cv.IMREAD_GRAYSCALE)
# blur = cv.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--profile', help='name of profile')
    parser.add_argument('-d', '--dataset', help='name of dataset directory to process images of')
    parser.add_argument('-pr', '--process', help='specify image processing technique')
    args = vars(parser.parse_args())

    return args


def get_input():
    args = get_args()
    profile, dataset, process = args.get('profile'), args.get('dataset'), args.get('process')
    return profile, dataset, process


def proc_images(root_path, process):

    image_dirs = glob.glob(f"{root_path}/*.png")
    for idx, directory in enumerate(image_dirs):
        image_dirs[idx] = directory.replace("\\", "/")

    for img in image_dirs:
        current_img = cv.imread(img, cv.IMREAD_GRAYSCALE)

        if process == "amt" or "adaptive mean threshold":
            current_img = cv.medianBlur(current_img, 5)
            current_img = cv.adaptiveThreshold(current_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        if process == "canny":
            pass
        if process == "clahe":
            pass
        if process == "otsu":
            pass
        cv.imwrite(img, current_img)


if __name__ == "__main__":
    profile, dataset, process = get_input()
    root_path = f"../../profiles/{profile}/data/{dataset}/all"

    try:
        proc_images(root_path, process)

    except FileNotFoundError:
        print("specify a valid dataset using the --profile and --dataset flags")
