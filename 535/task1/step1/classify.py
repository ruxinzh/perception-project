# fit a mask rcnn on the kangaroo dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os,sys
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

# class that defines and loads the kangaroo dataset
def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)

def get_bbox(p0, p1):

    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e
class KangarooDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir):
        # define one class
        self.add_class("dataset", 1, "kangaroo")
        # define data locations
        files = glob(dataset_dir+'/*/*_image.jpg')
        # find all images
        for idx in range(0,len(files)):
            # extract image id
            image_path=files[idx].split('\\')[-1]
            image_id=image_path.split('.')[-2]
            # skip bad images
            img_path = files[idx]
            ann_path = files[idx]
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file


    def extract_boxes(self, filename):
    #filename is trainval/*/*jpg
        xyz = np.fromfile(filename.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
        xyz = xyz.reshape([3, -1])
        proj = np.fromfile(filename.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
        proj.resize([3, 4])

        try:
            bbox = np.fromfile(filename.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
        except FileNotFoundError:
            print('[*] bbox not found.')
            bbox = np.array([], dtype=np.float32)

        bbox = bbox.reshape([-1, 11])

        uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
        uv = uv / uv[2, :]
        boxes = list()
        for k, b in enumerate(bbox):
            R = rot(b[0:3])
            t = b[3:6]
            sz = b[6:9]
            vert_3D, edges = get_bbox(-sz / 2, sz / 2)
            vert_3D = R @ vert_3D + t[:, np.newaxis]

            vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
            vert_2D = vert_2D / vert_2D[2, :]
        
            ignore_in_eval = bool(b[10])
            if ignore_in_eval:
                continue
            else:
                xmax=1914
                ymax=1052
                xl=int(min(vert_2D[0]))
                xr=int(max(vert_2D[0]))
                yb=int(min(vert_2D[1]))
                yu=int(max(vert_2D[1]))
                xmin=max(xl,0)
                xmax=min(xr,xmax)
                ymin=max(yb,0)
                ymax=min(yu,ymax)
                coors = [xmin, ymin, xmax, ymax]
                boxes.append(coors)
        width = int(1914)
        height = int(1052)
        return boxes, width, height
    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('kangaroo'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# define a configuration for the model
class KangarooConfig(Config):
    # define the name of the configuration
    NAME = "kangaroo_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 444

# prepare train set
train_set = KangarooDataset()
train_set.load_dataset('train4')
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# prepare test/val set
test_set = KangarooDataset()
test_set.load_dataset('test')
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
config = KangarooConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir='./', config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights('mask_rcnn_kangaroo_cfg_1129.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=0.01, epochs=1, layers='heads')
