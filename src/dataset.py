import torch
import zipfile
import xml.etree.ElementTree as ET
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


# class to idx in VOC dataset
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

CLS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}

class VOCDataset(Dataset):
    def __init__(self, img_paths_list, transform=None):
        self.img_paths_list = img_paths_list
        self.transform = transform

    def __len__(self):
        return len(self.img_paths_list)
    
    def __getitem__(self, idx):
        img_path = self.img_paths_list[idx]
        annotation_path = img_path.parents[1] / "Annotations" / (img_path.stem + ".xml")

        img = Image.open(img_path)
        annotation_dict = voc_to_dict(annotation_path)

        if self.transform:
            img, annotation_dict = self.transform(img, annotation_dict)
        
        return img, annotation_dict

def collate_fn(batch):
    imgs, annotations = zip(*batch)

    max_height = max(img.shape[1] for img in imgs)
    max_width = max(img.shape[2] for img in imgs)

    padded_imgs = []
    img_sizes_before_pad = []
    gt_boxes = []
    gt_labels = []

    for img, annotation in zip(imgs, annotations):
        img_height, img_width = img.shape[1], img.shape[2]
        pad_height = max_height - img_height
        pad_width = max_width - img_width

        img_sizes_before_pad.append((img_height, img_width))  # Store original height and width

        padded_img = nn.functional.pad(img, (0, pad_width, 0, pad_height), mode='constant', value=0)
        padded_imgs.append(padded_img)

        boxes = torch.tensor([
            [obj["bndbox"]["x_min"]* img_width, 
             obj["bndbox"]["y_min"]* img_height, 
             obj["bndbox"]["x_max"]* img_width, 
             obj["bndbox"]["y_max"]* img_height]
             for obj in annotation["objects"]], dtype=torch.float32)
        
        labels = torch.tensor([obj["class_idx"] for obj in annotation["objects"]], dtype=torch.int64)

        gt_boxes.append(boxes)
        gt_labels.append(labels)

    return torch.stack(padded_imgs, dim=0), gt_boxes, gt_labels, img_sizes_before_pad

def create_voc_dataloader(img_paths_list, transform=None, batch_size=32, shuffle=True):
    voc_dataset = VOCDataset(img_paths_list, transform=transform)
    dataloader = torch.utils.data.DataLoader(voc_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

# Converting VOC annotation XML to dictionary format
def voc_to_dict(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    img_width = float(root.find("size").find("width").text)
    img_height = float(root.find("size").find("height").text)

    annotation_data = {
        "filename" : root.find("filename").text,
        "size" : {
            "width" : img_width,
            "height" : img_height,
            "depth" : int(root.find("size").find("depth").text)
        },
        "objects" : []
    }

    for obj in root.findall("object"):

        name = obj.find("name").text
        obj_dict = {
            "name" : name,
            "class_idx" : CLS_TO_IDX[name],
            "bndbox" : {
                "x_min" : float(obj.find("bndbox").find("xmin").text)/img_width,
                "y_min" : float(obj.find("bndbox").find("ymin").text)/img_height,
                "x_max" : float(obj.find("bndbox").find("xmax").text)/img_width,
                "y_max" : float(obj.find("bndbox").find("ymax").text)/img_height
            }
        }

        annotation_data["objects"].append(obj_dict)

    return annotation_data

def get_voc_img_paths_train():
    data_path = Path("data/")
    if not data_path.exists():
        raise RuntimeError("Data directory not found. Please run data_setup.py to extract the datasets.")
    
    # Define the paths to the VOC2007 and VOC2012 datasets and their image directories
    voc2007_path = data_path / "VOC2007"
    voc2012_path = data_path / "VOC2012"

    voc2007_img_path = voc2007_path / "JPEGImages"
    voc2012_img_path = voc2012_path / "JPEGImages"

    voc2007_trainval = voc2007_path / "ImageSets" / "Layout" / "trainval.txt"
    voc2012_trainval = voc2012_path / "ImageSets" / "Layout" / "trainval.txt"

    with open(voc2007_trainval, "r") as f:
        voc2007_img_paths_train = [voc2007_img_path / (line.strip() + ".jpg") for line in f.readlines()]
    with open(voc2012_trainval, "r") as f:
        voc2012_img_paths_train = [voc2012_img_path / ((line.strip()).split(" ")[0] + ".jpg") for line in f.readlines()]

    return voc2007_img_paths_train, voc2012_img_paths_train

def get_voc_img_paths_test():
    data_path = Path("data/")
    if not data_path.exists():
        raise RuntimeError("Data directory not found. Please run data_setup.py to extract the datasets.")

    # Define the paths to the VOC2007 and VOC2012 datasets and their image directories
    voc2007_path = data_path / "VOC2007"
    voc2012_path = data_path / "VOC2012"

    voc2007_img_path = voc2007_path / "JPEGImages"
    voc2012_img_path = voc2012_path / "JPEGImages"

    voc2007_test = voc2007_path / "ImageSets" / "Layout" / "test.txt"
    voc2012_test = voc2012_path / "ImageSets" / "Layout" / "test.txt"

    with open(voc2007_test, "r") as f:
        voc2007_img_paths_test = [voc2007_img_path / (line.strip() + ".jpg") for line in f.readlines()]
    with open(voc2012_test, "r") as f:
        voc2012_img_paths_test = [voc2012_img_path / ((line.strip()).split(" ")[0] + ".jpg") for line in f.readlines()]

    return voc2007_img_paths_test, voc2012_img_paths_test

if __name__ == "__main__":
    data_path = Path("data/")

    # Get the paths to the VOC2007 and VOC2012 zip files
    voc2007_zip_path = Path("VOC2007.zip")
    voc2012_zip_path = Path("VOC2012.zip")

    # Check if the zip files exist, if not raise an error
    if not voc2007_zip_path.exists() or not voc2012_zip_path.exists():
        raise RuntimeError("Dataset not found.")


    # Extract the datasets in the data directory
    print("Extracting 2007 dataset ...")
    with zipfile.ZipFile(voc2007_zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    print("Extracting 2012 dataset ...")
    with zipfile.ZipFile(voc2012_zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)