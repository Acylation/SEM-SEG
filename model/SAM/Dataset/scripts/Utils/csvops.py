# csv文件常用操作
import csv

def create_csv(arg_path, header):
    file = open(arg_path, 'w', encoding= 'utf-8', newline='')
    csv.writer(file).writerow(header)
    file.close()

def write_csv(arg_path, row):
    file = open(arg_path, 'a+', encoding= 'utf-8', newline='')
    csv.writer(file).writerow(row)
    file.close()

def read_csv(arg_path):
    file = open(arg_path, 'r', encoding= 'utf-8', newline='')
    for line in csv.reader(file):
        print(line)
    file.close()