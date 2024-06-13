import os
import argparse
import yaml
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def yolo_to_voc(yolo_label_path, img_width, img_height):
    with open(yolo_label_path, 'r') as file:
        data = file.readlines()

    voc_data = []
    for dt in data:
        # Split string to float
        c, x, y, w, h = map(float, dt.split())

        # Convert YOLO format to VOC format
        l = int((x - w / 2) * img_width)
        r = int((x + w / 2) * img_width)
        t = int((y - h / 2) * img_height)
        b = int((y + h / 2) * img_height)

        # Ensure coordinates are within image boundaries
        l = max(0, l)
        r = min(img_width - 1, r)
        t = max(0, t)
        b = min(img_height - 1, b)

        voc_data.append(f"{int(c)} {l} {t} {r} {b}\n")

    with open(yolo_label_path, 'w') as file:
        file.writelines(voc_data)

def convert_labels(split, dataroot, img_width, img_height):
    label_dir = os.path.join(dataroot, split, 'labels')
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        yolo_to_voc(label_path, img_width, img_height)

def convert_all_labels(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    dataroot = config['dataroot']
    img_width = config['img_width']
    img_height = config['img_height']

    splits = ['train', 'valid', 'test']
    num_splits = len(splits)

    # Determine number of processes to use (cores available)
    num_processes = min(cpu_count(), num_splits)

    # Create a pool of processes
    with Pool(num_processes) as pool:
        pool.starmap(convert_labels, [(split, dataroot, img_width, img_height) for split in splits])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert YOLO format labels to VOC format using multiprocessing.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')

    args = parser.parse_args()
    convert_all_labels(args.config)
