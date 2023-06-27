import os
from xml.etree import ElementTree as et


def extract_bbox_xml(img_dir):
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
