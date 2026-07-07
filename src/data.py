from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

# class to idx in VOC dataset
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

CLS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}

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
        xmin = float(obj.find("bndbox").find("xmin").text)
        ymin = float(obj.find("bndbox").find("ymin").text)
        xmax = float(obj.find("bndbox").find("xmax").text)
        ymax = float(obj.find("bndbox").find("ymax").text)

        name = obj.find("name").text
        obj_dict = {
            "name" : name,
            "class_idx" : CLS_TO_IDX[name],
            "bndbox" : {
                "x_c" : ((xmin + xmax) / 2)/img_width,
                "y_c" : (ymin + ymax) / 2/img_height,
                "width" : (xmax - xmin)/img_width,
                "height" : (ymax - ymin)/img_height
            }
        }

        annotation_data["objects"].append(obj_dict)

    return annotation_data

def get_voc_img_paths(data_path):
    # Define the paths to the VOC2007 and VOC2012 datasets and their image directories
    voc2007_path = data_path / "VOC2007"
    voc2012_path = data_path / "VOC2012"

    voc2007_img_path = voc2007_path / "JPEGImages"
    voc2012_img_path = voc2012_path / "JPEGImages"

    voc2007_img_paths_list = list(voc2007_img_path.glob("*.jpg"))
    voc2012_img_paths_list = list(voc2012_img_path.glob("*.jpg"))

    return voc2007_img_paths_list, voc2012_img_paths_list

if __name__ == "__main__":
    # Create data directory if it doesn't exist and store its path
    data_path = Path("data/")
    data_path.mkdir(exist_ok=True)

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

