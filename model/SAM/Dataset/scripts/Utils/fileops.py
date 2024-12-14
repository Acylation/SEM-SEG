import os

# for loading files

def validate_path(path):
    if not os.path.exists(path):
        raise ValueError("Error: invalid path, please check your input. {path}".format(path = path))

def get_path_type(path):
    validate_path(path)
    if(os.path.isfile(path)):
        return 'file'
    if(os.path.isdir(path)):
        return 'folder'
    
def get_base_name(path):
    '''file name with extension'''
    if get_path_type(path) == 'file':
        basename = os.path.basename(path)
        return basename
    if get_path_type(path) == 'folder':
        raise ValueError("Error: Can't get a folder's basename, please check your input. {path}".format(path = path))

def get_file_name(path):
    '''file name without extension'''
    if get_path_type(path) == 'file':
        basename = os.path.basename(path)
        name = os.path.splitext(basename)[0]
        return name
    if get_path_type(path) == 'folder':
        raise ValueError("Error: Can't get a folder's filename, please check your input. {path}".format(path = path))

def get_str_name(path_str):
    '''split name from a filename-like string, without validation'''
    return os.path.splitext(path_str)[0]

def get_ext(path):
    if get_path_type(path) == 'file':
        basename = os.path.basename(path)
        extension = os.path.splitext(basename)[1]
        return extension
    if get_path_type(path) == 'folder':
        raise ValueError("Error: Can't get a folder's extension, please check your input. {path}".format(path = path))
    return extension

def print_load_report(path, list):
    '''print loading report in the terminal'''
    if len(list) == 0:
        print("Warning: return empty list. {path}".format(path = path))
    print("From {path} imported {n} items".format(path = path, n = len(list)))

# for saving files

def get_or_create_folder(path):
    if os.path.exists(path):
        # print("{path} found.".format(path = path))
        return path
    else:
        print("{path} not exists, it's now created.".format(path = path))
        os.makedirs(path)
        return path