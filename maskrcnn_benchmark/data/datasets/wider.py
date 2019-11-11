import os

import torch
import torch.utils.data
from PIL import Image
import sys
import scipy.io as sio
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList


class WiderFaceDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "face",
    )

    def __init__(self, data_dir, transforms=None):
        self.root = data_dir
        self.transforms = transforms
        self._annopath = os.path.join(self.root, 'annotations', '%s')
        self._imgpath = os.path.join(self.root, 'WIDER_train/images', '%s')
        self._imgsetpath = os.path.join(self.root, "img_list.txt")
        self.ids = list()
        # it's a list of tuples: [(*.jpg, *.xml), ()...]
        with open(self._imgsetpath, 'r') as f:
          self.ids = [tuple(line.strip("\n").split()) for line in f]

        cls = WiderFaceDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id[0]).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id[1]).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        # target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            # difficult = int(obj.find("difficult").text) == 1
            # if not self.keep_difficult and difficult:
            #    continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            # difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            # "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id[1]).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return WiderFaceDataset.CLASSES[class_id]


class WiderFaceTestDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "face",
    )

    def __init__(self, data_dir, transforms=None):
        self.root = data_dir
        self.transforms = transforms
        self._annopath = os.path.join(self.root, 'wider_face_split/wider_face_val.mat')
        self._imgpath = os.path.join(self.root, 'WIDER_val/images', '%s')
        self._imginfo = os.path.join(self.root, 'WIDER_val_info.txt')
        wider_face = sio.loadmat(self._annopath)
        event_list = wider_face['event_list']
        file_list = wider_face['file_list']
        self.ids = list()
        with open(self._imginfo, "r") as f:
            self.imginfo = f.readlines()
        for index, event in enumerate(event_list):
            filelist = file_list[index][0]
            for num, file in enumerate(filelist):
                self.ids.append((event[0][0], file[0][0]))

        cls = WiderFaceDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        sub_path = os.path.join(img_id[0], img_id[1]) + ".jpg"
        img = Image.open(self._imgpath % sub_path).convert("RGB")
        # fake target
        target = BoxList([[10,10,30,30]],(600,600))
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        # print("Test get_img_info:{}".format(index))
        img_id = self.ids[index]
        height, width = self.imginfo[index].strip("\n").split()
        return {"event": img_id[0], "name": img_id[1], "height": height, "width": width}

    def map_class_id_to_class_name(self, class_id):
        return WiderFaceTestDataset.CLASSES[class_id]
