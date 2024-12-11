import sys
import json
import numpy as np
import pandas as pd
import torch
import torchvision

def main():
    # Check if exactly two arguments are passed
    if len(sys.argv) != 4:
        print(json.dumps({"error": "Three arguments are required"}))
        return

    str1, str2, file_path = sys.argv[1], sys.argv[2], sys.argv[3]
    # str1 = "gen"
    # str2 = "shin"
    # file_path = "./resources/backend/test.txt"
    result = str1 + str2

    # Output the result as JSON
    print(json.dumps({"result": result, "np version": np.__version__, "pd version": pd.__version__, "torch version": torch.__version__, "tv version": torchvision.__version__}))
    
    # works for dev
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(script_dir, "test.txt")
    
    if("app.asar" in file_path):
        file_path = file_path.replace("app.asar", "app.asar.unpacked")
    
    try:
        # Attempt to open the file and read its content
        with open(file_path, "r") as file:
            content = file.read()
        print(json.dumps({"file content": content}))
    except Exception as e:
        print(json.dumps({"file error": f"Unable to read file: {str(e)}"}))

if __name__ == "__main__":
    main()