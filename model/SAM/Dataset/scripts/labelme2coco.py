# labelme to COCO 
# Directly uses code from the labelme repository
# Adapted by ChatGPT to keep the sample name
# Prompt: Adapt this code, to keep the original sample name <code>

import collections
import datetime
import uuid
import os
import glob
from Utils.fileops import get_or_create_folder
from Utils.jsonops import save_json
import labelme
import pycocotools.mask
import cv2
import numpy as np

def convert(input_dir, output_dir, labels=['__ignore__', '_background_', 'sphere']):
    get_or_create_folder(output_dir)
    print("Creating dataset:", output_dir)

    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[
            dict(
                url=None,
                id=0,
                name=None,
            )
        ],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(labels):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(
                supercategory=None,
                id=class_id,
                name=class_name,
            )
        )

    out_ann_file = os.path.join(output_dir, "annotations.json")
    label_files = glob.glob(os.path.join(input_dir, "*.json"))

    for image_id, filename in enumerate(label_files):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)
        original_name = os.path.splitext(os.path.basename(label_file.imagePath))[0]
        ext = os.path.splitext(label_file.imagePath)[1]

        out_img_file = os.path.join(
            get_or_create_folder(os.path.join(output_dir, 'raw')), original_name + ext
        )

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        if not os.path.exists(out_img_file):
            cv2.imwrite(out_img_file, img)

        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=os.path.relpath(out_img_file, os.path.dirname(out_ann_file)).replace("\\", "/"), # Specially for linux platform
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            if shape_type == "circle":
                (x1, y1), (x2, y2) = points
                r = np.linalg.norm([x2 - x1, y2 - y1])
                n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                i = np.arange(n_points_circle)
                x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                points = np.stack((x, y), axis=1).flatten().tolist()
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))

            # START MODIFICATION
            # Prompt: I also want to plant bbox expanding , as I want to prompt a model based on bbox to learn.
            #   Rather thant do `bbox = pycocotools.mask.toBbox(mask).flatten().tolist()`, here I have a piece of code adding some bias. 
            #   Please adapt: <my original code, takeing all the math parts>
            # Modify bbox calculation to include a bias
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()
            x, y, w, h = bbox
            bias = 0.025
            x1 = x + np.random.normal(-bias * w, bias / 3 * w)
            y1 = y + np.random.normal(-bias * h, bias / 3 * h)
            x2 = x1 + w + np.random.normal(bias * w, bias / 3 * w)
            y2 = y + h + np.random.normal(bias * h, bias / 3 * h)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1] - 1, x2)
            y2 = min(img.shape[0] - 1, y2)

            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            # END MODIFICATION

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

    save_json(out_ann_file, data)

script_dir = os.path.dirname(os.path.abspath(__file__))

input_folder = os.path.join(script_dir, "../Train_raw")
output_folder = os.path.join(script_dir, "../Train")
convert(input_folder, output_folder)

input_folder = os.path.join(script_dir, "../Test_raw")
output_folder = os.path.join(script_dir, "../Test")
convert(input_folder, output_folder)
