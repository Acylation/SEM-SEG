import glob
import json

from Utils.fileops import get_path_type, get_ext, print_load_report

def load_json(arg_path):
    annlist = []
    if get_path_type(arg_path) == 'file' and get_ext(arg_path) == ".json":
        with open(arg_path, "r", encoding="utf-8") as f:
            annlist.append(json.load(f))
    if get_path_type(arg_path) == 'folder':
        paths = glob.glob(arg_path+"\\*"+".json")
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                annlist.append(json.load(f))
    print_load_report(arg_path, annlist)
    return annlist

def save_json(path, ann):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=4)