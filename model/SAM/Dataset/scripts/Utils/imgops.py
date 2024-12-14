EXTS = [".png", ".bmp", ".jpg"] # 读取文件夹时包含的文件类型

import os, sys
sys.path.append(os.getcwd())

import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from Utils.fileops import get_path_type, print_load_report, get_ext

def load_img(arg_path):
    '''load 3 channel BGR images from a single file or folder, return matlist as Mat[]'''
    matlist = []
    if(get_path_type(arg_path) == 'file'):
        if get_ext(arg_path) in EXTS:
            matlist.append(cv2.imread(arg_path))
        else:
            raise ValueError("Invalid extension for images, please check your input. {path}".format(path = arg_path))
    if(get_path_type(arg_path) == 'folder'):
        for EXT in EXTS:
            paths = glob.glob(arg_path+"\\*"+EXT)
            for path in paths:
                matlist.append(cv2.imread(path))
    print_load_report(arg_path, matlist)
    return matlist

def load_img_gray(path):
    '''load gray scale images from a single file or folder, return matlist as Mat[]'''
    matlist = []
    if(get_path_type(path) == 'file'):
        matlist.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    if(get_path_type(path) == 'folder'):
        for EXT in EXTS:
            files = glob.glob(path+"\\*"+EXT)
        for file in files:
            matlist.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
    print_load_report(path, matlist)
    return matlist

# def save():

def merge(img1, img2):
    fusion = cv2.addWeighted(img1, 1, img2, 1, 0)
    return fusion

def batchMerge(imglist):
    fusion = np.zeros([1024, 1280, 3], np.uint8) # 全局 shape 设定
    for img in imglist:
        fusion = merge(fusion, img)
    return fusion

def overlay(base, mask, alpha=0.5):
    print(base.shape)
    print(mask.shape)
    fusion = cv2.addWeighted(base, 1, mask, alpha, 0)
    return fusion

def roimask(base, mask):
    background = np.zeros([1024, 1280, 3], np.uint8)
    fusion = cv2.add(base, background, mask = mask)
    return fusion

def copyTo(base, front, mask):
    h, w, c = base.shape
    dst = np.zeros([h,w,c], np.uint8)

    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    front_gray = cv2.cvtColor(front, cv2.COLOR_BGR2GRAY)

    base_array = np.uint8(np.asarray(base_gray).flatten())
    front_array = np.uint8(np.asarray(front_gray).flatten())

    mask_index = np.bool_(np.asarray(mask).flatten())

    base_array[mask_index] = front_array[mask_index]
    dst_gray = base_array.reshape(h, w)
    dst = cv2.cvtColor(dst_gray, cv2.COLOR_GRAY2BGR)
    
    # 超低效实现
    # for row in range(h):
    #     for col in range(w):
    #         if mask[row, col] != 0:
    #             dst[row, col] = front[row, col]
    #         elif mask[row, col] == 0:
    #             dst[row, col] = base[row, col]
    
    return dst

def display(img):
    plt.imshow(img)
    plt.show()

# draw

def drawpoly(points, color = 255):
    background = np.zeros([1024, 1280, 3], np.uint8)
    points = np.array(points, np.int32)
    mask = cv2.fillPoly(background, [points], (color, color, color), lineType=cv2.LINE_4)
    return mask