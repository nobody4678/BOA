
import sys
sys.path.append('.')
import functools
import glob
import xml

import PIL
import cv2
import numpy as np
from tqdm import tqdm
import scipy.misc as m

from utils.augmentation import improc
# import paths
from utils.augmentation import util
DATA_ROOT = '/data/syguan/other_datasets'

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
        Returns:
            np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
        )

def encode_mask(mask):
    """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


@functools.lru_cache()
@util.cache_result_on_disk(
    f'{DATA_ROOT}/cache/pascal_voc_occluders.pkl', min_time="2020-9-5T10:12:32")
def load_occluders():
    image_mask_pairs = []
    pascal_root = f'{DATA_ROOT}/pascal_voc'
    image_paths = []
    for annotation_path in tqdm(glob.glob(f'{pascal_root}/Annotations/*.xml')):
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = (xml_root.find('segmented').text != '0')

        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall('object')):
            is_person = (obj.find('name').text == 'person')
            is_difficult = (obj.find('difficult').text != '0')
            is_truncated = (obj.find('truncated').text != '0')
            if not is_person and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))
        
        if not boxes:
            continue

        image_filename = xml_root.find('filename').text
        segmentation_filename = image_filename.replace('jpg', 'png')

        path = f'{pascal_root}/JPEGImages/{image_filename}'
        seg_path = f'{pascal_root}/SegmentationObject/{segmentation_filename}'

        im = improc.imread_jpeg(path)
        # labels = m.imread(seg_path) #np.array(PIL.Image.open(seg_path).convert('L')) 
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            # import ipdb
            # ipdb.set_trace()
            # object_mask = encode_mask(labels[ymin:ymax, xmin:xmax]).astype(np.uint8) 
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8)
            object_image = im[ymin:ymax, xmin:xmax]
            # Ignore small objects
            if cv2.countNonZero(object_mask) < 500:
                continue
            # import ipdb; ipdb.set_trace()

            object_mask = soften_mask(object_mask)
            downscale_factor = 0.5
            object_image = improc.resize_by_factor(object_image, downscale_factor)
            object_mask = improc.resize_by_factor(object_mask, downscale_factor)
            image_mask_pairs.append((object_image, object_mask))
            image_paths.append(path)

    return image_mask_pairs


def soften_mask(mask):
    morph_elem = improc.get_structuring_element(cv2.MORPH_ELLIPSE, (8, 8))
    eroded = cv2.erode(mask, morph_elem)
    result = mask.astype(np.float32)
    result[eroded < result] = 0.75
    return result


if __name__ == '__main__':
    util.cache_result_on_disk(f'{DATA_ROOT}/cache/pascal_voc_occluders.pkl', min_time="2018-11-08T15:13:38")
    load_occluders()