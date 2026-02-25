from typing import Dict
import os
import cv2
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import shutil

np.random.seed(42)

def create_yolo_structure(base_path="ycb_processed"):
    Path(f"{base_path}/images/train").mkdir(parents=True, exist_ok=True)
    Path(f"{base_path}/images/val").mkdir(parents=True, exist_ok=True)
    Path(f"{base_path}/labels/train").mkdir(parents=True, exist_ok=True)
    Path(f"{base_path}/labels/val").mkdir(parents=True, exist_ok=True)

def save_image_files(orig_folder:str,dest_folder:str,stems:np.array,folder_prefix:str):

    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    for stem in stems:
        orig = os.path.join(orig_folder, stem + '-color.jpg')

        # remove -color and prefix folder
        new_name = f"{folder_prefix}_{stem}.jpg"
        dest = os.path.join(dest_folder, new_name)

        if os.path.isfile(orig):
            shutil.copy(orig, dest)

    
def to_yolo_format(orig_folder: str,
                   dest_folder: str,
                   stems: np.array,
                   id_to_class: dict,
                   folder_prefix: str):
    
    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    for stem in stems:

        label_path = os.path.join(orig_folder, stem + '-label.png')
        if not os.path.isfile(label_path):
            continue

        labeled_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if labeled_img is None:
            continue

        h, w = labeled_img.shape
        unique_ids = np.unique(labeled_img)
        unique_ids = unique_ids[unique_ids != 0]

        yolo_lines = []

        for obj_id in unique_ids:
            if str(obj_id) not in id_to_class:
                continue

            class_index = id_to_class[str(obj_id)]
            mask = (labeled_img == obj_id).astype(np.uint8)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if len(contour) < 3:
                    continue
                if cv2.contourArea(contour) < 20:
                    continue

                contour = contour.reshape(-1, 2)

                polygon = []
                for point in contour:
                    x = point[0] / w
                    y = point[1] / h
                    polygon.extend([x, y])

                if len(polygon) >= 6:
                    line = str(class_index) + " " + " ".join(map(str, polygon))
                    yolo_lines.append(line)

        # Save with prefix
        txt_name = f"{folder_prefix}_{stem}.txt"
        txt_path = os.path.join(dest_folder, txt_name)

        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))



def create_yolo_train_dataset(data_path:str,val_frac:float=0.2) -> None:
    """
    Creates folder structure for Yolo Training from video raw structure dataset.

    Creates data directories for training
    """
    base_path = 'ycb_processed'
    Path(base_path).mkdir(exist_ok=True) # Creates outer folder

    create_yolo_structure(base_path)

    # Load uint8 label mask to class index dict
    with open('src/utils/id_idx_map.json') as json_file:
        id_to_class = json.load(json_file)

    # List all folders in data
    f_l = sorted(os.listdir(data_path))

    # For each folder, gather which files go to train, which to test

    for folder in tqdm(f_l):
        stem_files = [image.replace('-color.jpg','') for image in os.listdir(os.path.join(data_path,folder)) if image.endswith('-color.jpg')]
        val_size = max(1, int(val_frac * len(stem_files)))

        val_instances = np.random.choice(stem_files,size=val_size,replace=False)
        train_instances = np.setdiff1d(stem_files,val_instances,assume_unique=True)

        # Save image files 
        save_image_files(os.path.join(data_path,folder),
                        'ycb_processed/images/train',
                        train_instances,
                        folder)
        save_image_files(os.path.join(data_path,folder),
                 'ycb_processed/images/val',
                 val_instances,
                 folder)

        # Convert label to segmentation
        to_yolo_format(os.path.join(data_path,folder),
                    'ycb_processed/labels/train',
                    train_instances,
                    id_to_class,
                    folder)      
        to_yolo_format(os.path.join(data_path,folder),
                    'ycb_processed/labels/val',
                    val_instances,
                    id_to_class,
                    folder)

def build_votes(data_folder:str) -> Dict:
    votes = defaultdict(lambda: defaultdict(int))

    fu = sorted(os.listdir(data_folder))


    for f in tqdm.tqdm(fu):
        subdir = os.path.join(data_folder,f)
        boxes = sorted([b for b in os.listdir(subdir) if b.endswith('-box.txt')])

        for box in boxes:
            stem = box.replace('-box.txt', '')
            label = f"{stem}-label.png"

        
            l_img = cv2.imread(os.path.join(subdir,label),cv2.IMREAD_GRAYSCALE)
            if l_img is None or l_img.size ==0: continue
            bboxes = process_box_txt(os.path.join(subdir, box))

            for bbox in bboxes:        
                ymin,ymax,xmin,xmax = bbox['ymin'],bbox['ymax'],bbox['xmin'],bbox['xmax']

                if xmin < xmax and ymin < ymax:
                    h, w = l_img.shape



                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(w, xmax)
                    ymax = min(h, ymax)
                
                    region = l_img[ymin:ymax,xmin:xmax]

                    values, counts = np.unique(region, return_counts=True)

                    mask = values != 0
                    values = values[mask]
                    counts = counts[mask]

                    if values.size == 0:
                        continue
                    dominant_id = values[np.argmax(counts)]

                    votes[dominant_id][bbox['name']] +=1

    final_mapping = {}

    for seg_id, name_counts in votes.items():
        best_name = max(name_counts, key=name_counts.get)
        final_mapping[str(seg_id)] = best_name
    
    return final_mapping

def process_box_txt(p_box):
    res = []
    with open(p_box) as f:
        lines = [line.rstrip().split(' ') for line in f]
    
    for line in lines:
        res.append({'name':line[0],
                    'xmin':int(float(line[1])),
                    'ymin':int(float(line[2])),
                    'xmax':int(float(line[3])),
                    'ymax':int(float(line[4]))     
                    })
    return res




if __name__ == "__main__":
    create_yolo_train_dataset('data')










