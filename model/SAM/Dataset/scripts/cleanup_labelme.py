# validating and cropping labelme annotations
# Reuse of my previous code

import os
from Utils.fileops import get_str_name, get_or_create_folder
from Utils.jsonops import load_json, save_json

# A polygon should have more than three points
def validate_points(annotation):
    shapes = []
    for instance in annotation["shapes"]:
        points = instance["points"]
        if len(points) >= 3: # filter
            shapes.append(instance)
        else:
            print(f'A polygon must contain at lease 3 points. {points} in {annotation["imagePath"]}')
    annotation["shapes"] = shapes
    return annotation

# Crop annotation to ROI
def crop(annotation):
    for instance in annotation["shapes"]:
        points = instance["points"]
        for point in points:
            if point[1] > 959.0:
                point[1] = 959.0
    return annotation

def cvtImage(annotation):
    imagePath = annotation["imagePath"]
    imageData = annotation["imageData"]

script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_dir, "../annotations")
output_folder = os.path.join(script_dir, "../annotations_cropped")

labelme_anns = load_json(input_folder)
for labelme_ann in labelme_anns:
    labelme_ann = validate_points(labelme_ann)
    labelme_ann = crop(labelme_ann)
    save_json(os.path.join(get_or_create_folder(output_folder) , get_str_name(labelme_ann["imagePath"]) + ".json") , labelme_ann)
    