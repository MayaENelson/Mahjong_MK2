import PySimpleGUI as sg
import os
import importlib
from universal_fx import extract_bbox_xml
import cv2 as cv
import numpy as np
import glob
import random
import colorsys
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--labels', help='specify label set to import')
    args = vars(parser.parse_args())

    return args


def get_input():
    args = get_args()
    labels = args.get('labels')
    return labels


def instance_request(instance_name):
    module = importlib.import_module("labels")
    instance = getattr(module, instance_name)
    return instance


def random_bright_color():
    h = random.uniform(0, 1)  # Random hue value
    s = random.uniform(0.5, 1)  # Random saturation value (higher values for brighter colors)
    v = random.uniform(0.7, 1)  # Random value/brightness value (higher values for brighter colors)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


if __name__ == "__main__":
    label_type = get_input()
    labels = instance_request(label_type)
    del labels[0]
    labels.insert(0, "All")

    image = cv.imread("../../profiles/mahjong/data/NESW_raw/all/original_0.png")

    right_col = [[sg.Text('Dataset'), sg.In(size=(25, 1), enable_events=True, key='-FOLDER-'), sg.FolderBrowse()],
                 [sg.Listbox(values=[], enable_events=True, size=(40, 20), key='-FILE LIST-')],
                 [sg.Listbox(values=labels, enable_events=True, size=(40, 20), key='-LABELS-')],
                 [sg.Text('Image: ', visible=False, key='-IMAGE TEXT-'), sg.Text('', visible=False, key='-NUMERATOR TEXT-'),
                  sg.Text(' / ', visible=False, key='-SLASH TEXT-'), sg.Text('', visible=False, key='-DENOMINATOR TEXT-'),
                  sg.Button('◄', size=(3, 1), visible=False, key='-LEFT BUTTON-'),
                  sg.Button('►', size=(3, 1), visible=False, key='-RIGHT BUTTON-')]]

    left_col = [[sg.Image(key='-IMAGE-')],
                [sg.Text("Select a dataset sub-folder to start.", key="-INFO-")]]

    layout = [[sg.Column(left_col, element_justification='c'),
               sg.VSeperator(),
               sg.Column(right_col, element_justification='c')]]

    window = sg.Window('Annotation Viewer', layout)
    file_name = ""

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == '-FOLDER-':
            folder = values['-FOLDER-']
            try:
                file_list = os.listdir(folder)
            except OSError:
                file_list = []
            file_names = [file_name for file_name in file_list
                          if os.path.isfile(os.path.join(folder, file_name)) and file_name.lower().endswith(".png")]
            window['-FILE LIST-'].update(file_names)
            window['-INFO-'].update('')
        if event == '-FILE LIST-':
            selected_item = values['-FILE LIST-']
            if selected_item:
                file_name = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
                image = cv.imread(file_name)

                list_files = glob.glob(f"{values['-FOLDER-']}/*.png")
                for idx, directory in enumerate(list_files):
                    list_files[idx] = directory.replace("\\", "/")

                num_files = len(list_files)
                file_to_find = f"{values['-FOLDER-']}/{values['-FILE LIST-'][0]}"
                file_idx = list_files.index(file_to_find)

                window['-IMAGE-'].update(data=cv.imencode('.png', image)[1].tobytes())
                window['-IMAGE TEXT-'].update(visible=True)
                window['-NUMERATOR TEXT-'].update(visible=True, value=file_idx+1)
                window['-SLASH TEXT-'].update(visible=True)
                window['-DENOMINATOR TEXT-'].update(visible=True, value=num_files)
                window['-LEFT BUTTON-'].update(visible=True)
                window['-RIGHT BUTTON-'].update(visible=True)
        if event == "-LABELS-":
            selected_item = values['-LABELS-'][0]
            try:
                boxes, labels = extract_bbox_xml(f"{file_name.split('.')[0]}.xml")
                image_copy = np.copy(image)
                if selected_item == 'All':
                    label_colors = {}
                    for label in labels:
                        color = random_bright_color()
                        label_colors[label] = color
                    for idx, label in enumerate(labels):
                        xmin = int(boxes[idx][0])
                        ymin = int(boxes[idx][1])
                        xmax = int(boxes[idx][2])
                        ymax = int(boxes[idx][3])
                        color = label_colors[label]
                        cv.rectangle(image_copy, (xmin, ymin), (xmax, ymax), color, 2)

                        text_x = boxes[idx][0]
                        text_y = boxes[idx][1]
                        cv.putText(image_copy, label, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, fontScale=.5,
                                   color=(255, 255, 255), thickness=2)

                for idx, label in enumerate(labels):
                    if label == selected_item:

                        xmin = int(boxes[idx][0])
                        ymin = int(boxes[idx][1])
                        xmax = int(boxes[idx][2])
                        ymax = int(boxes[idx][3])

                        cv.rectangle(image_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                window['-INFO-'].update('')
                window['-IMAGE-'].update(data=cv.imencode('.png', image_copy)[1].tobytes())
            except FileNotFoundError:
                window['-INFO-'].update('XML not available.')
        if event == "-LEFT BUTTON-":
            list_files = glob.glob(f"{values['-FOLDER-']}/*.png")
            for idx, directory in enumerate(list_files):
                list_files[idx] = directory.replace("\\", "/")

            num_files = len(list_files)
            file_to_find = f"{values['-FOLDER-']}/{values['-FILE LIST-'][0]}"
            current_idx = list_files.index(file_to_find) - 1
            if current_idx < 0:
                current_idx = 0

            window['-FILE LIST-'].update(set_to_index=current_idx)
            window['-IMAGE-'].update(filename=list_files[current_idx])
            window['-NUMERATOR TEXT-'].update(value=current_idx+1)
        if event == "-RIGHT BUTTON-":
            list_files = glob.glob(f"{values['-FOLDER-']}/*.png")
            for idx, directory in enumerate(list_files):
                list_files[idx] = directory.replace("\\", "/")

            num_files = len(list_files)
            file_to_find = f"{values['-FOLDER-']}/{values['-FILE LIST-'][0]}"
            current_idx = list_files.index(file_to_find) + 1
            if current_idx >= num_files:
                current_idx = num_files-1

            window['-FILE LIST-'].update(set_to_index=current_idx)
            window['-IMAGE-'].update(filename=list_files[current_idx])
            window['-NUMERATOR TEXT-'].update(value=current_idx + 1)
    window.close()