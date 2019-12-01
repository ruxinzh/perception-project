# fit a mask rcnn on the kangaroo dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import numpy as np
import cv2
from PIL import Image
import os,sys
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes


from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.model import mold_image
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
        files = glob(dataset_dir+'/*_image.jpg')
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
    STEPS_PER_EPOCH = 94
# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "kangaroo_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP    
def plot_actual_vs_predicted(image,mask, model, cfg, n_images=1):
    # load image and mask
        # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
        # convert image into one sample
    sample = expand_dims(scaled_image, 0)
        # make prediction
    yhat = model.detect(sample, verbose=0)[0]
        # define subplot
    pyplot.subplot(n_images, 2, 1)
        # plot raw pixel data
    pyplot.imshow(image)
    pyplot.title('Actual')
        # plot masks
    for j in range(mask.shape[2]):
        pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
    pyplot.subplot(n_images, 2, 2)
        # plot raw pixel data
    pyplot.imshow(image)
    pyplot.title('Predicted')
    ax = pyplot.gca()
        # plot each box
#    for box,i in zip(yhat['rois'],range(len(yhat['rois']))):
            # get coordinates
    score=yhat['scores']
    index=np.argmax(score)
    box=yhat['rois'][index]
    y1, x1, y2, x2 = box
            # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
            # create the shape
    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
    display_txt = '{:0.2f}, car'.format(yhat['scores'][index])
    ax.add_patch(rect)
    ax.text(x1, y1, display_txt, bbox={'facecolor':'red', 'alpha':0.5})
    # show the figure
    pyplot.show()

def bboxcut (image,image_path, model, cut_path, cfg):    
    scaled_image = mold_image(image, cfg)
    sample = expand_dims(scaled_image, 0)
    yhat = model.detect(sample, verbose=0)[0]
    score=yhat['scores']
    if not score.size:
        lb=0
    else:
        index=np.argmax(score)
        box=yhat['rois'][index]
        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1
        xmax=1914
        ymax=1052
        xl=x1
        xr=x2
        yb=y1
        yu=y2
        xl=max(xl,0)
        xr=min(xr,xmax)
        yb=max(yb,0)
        yu=min(yu,ymax)
        file_name=image_path.split('/')[-1]
        folder_name=cut_path+image_path.split('/')[-2]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        if score[index]>=0.95:
            img1=Image.open(image_path)
            imgNew=img1.crop((xl,yb,xr,yu))
            imgNew.save(os.path.join(folder_name,os.path.basename(file_name)))
            lb=1
        else:
            lb=0
    return lb
    

# prepare train set


# prepare test/val set
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('mask_rcnn_kangaroo_cfg_0004.h5', by_name=True)
#train_mAP = evaluate_model(train_set, model, cfg)
#print("Train mAP: %.3f" % train_mAP)
# plot predictions for train dataset

folder_lj=glob('test/*')    #change dir
path_test = 'test/*/*.jpg'
files_test = glob(path_test)
label=np.zeros((len(files_test),1))
count=0
for j in range(len(folder_lj)):
    folder_name=folder_lj[j].split('\\')[-1]
    train_set = KangarooDataset()
    train_folder_path='test/'+folder_name
    train_set.load_dataset(train_folder_path)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    test_img=glob(train_folder_path+'/*.jpg') #change dir
    for i in range(len(test_img)):
        snapshot = test_img[i]
        #guid=snapshot.split('\\')[-2]
        #guid=guid.split('/')[-1]
        idx=snapshot.split('\\')[-1]
        cut_path = 'car/'
        image_path='{}/{}'.format(train_folder_path,idx)
        image = train_set.load_image(i)
        label[count+i]=bboxcut(image,image_path, model, cut_path, cfg)
    count=count+ len(test_img)
names =  []
for k in range(len(files_test)):
    snapshot1 = files_test[k]
    guid=snapshot1.split('\\')[-2]
    guid=guid.split('/')[-1]
    idx1=snapshot1.split('\\')[-1]
    idx1=idx1.split('.')[-2]
    idx1=idx1.split('_')[-2]
    names.append(str(guid+"/"+idx1))

labels = label.astype(int)      
names = np.reshape(np.array(names),(len(files_test),1))
final = np.column_stack((names,labels))
np.savetxt('car1129.csv', final, delimiter=',', header = "guid/image,label",fmt='%s',comments ='')


