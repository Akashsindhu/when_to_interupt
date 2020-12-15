import pandas as pd
import numpy as np
import cv2
import os
from glob import glob
import json
from shutil import copy

labels = "../labels/"


if __name__=="__main__":

    labels_lst = []
    image_file_lst = []
    video_folder_name = []
    for file in glob(labels + "*.json"):
        with open(file, mode='r') as writer:
            text = writer.read()
            res = json.loads(text)
            # print(file)

            frames_directory = res['framesDirectory']
            labels_each_video = res['labels']
            labels_lst.append(labels_each_video)
            # print(frames_directory)
            # print(labels_each_video)
            path_name_lst = frames_directory.split("\\")
            video_folder_name.append(path_name_lst[1])
            for file in glob(os.path.join("../" + path_name_lst[0] + "/" + path_name_lst[1] + "/")):
                image_file_lst.append(file)
                # print(file)
        # print(len(labels_lst))
        for image_dir, label in zip(image_file_lst, labels_lst):
            print(image_dir)
            # print(label)
            image_lst = os.listdir(image_dir)
            for image, key in zip(image_lst, label):
                # print(image)
                # print(label[key])

                if not os.path.exists("../images/" + os.path.join(label[key])): #  if directory not exists
                    os.mkdir("../images/" + os.path.join(label[key])) # create directory
                if not os.path.exists("../images/" + os.path.join(label[key]) + "/" + path_name_lst[1]): # if subdirectory not exists
                    os.mkdir("../images/" + os.path.join(label[key]) + "/" + path_name_lst[1])
                    copy(os.path.join(image_dir, image), "../images/" + os.path.join(label[key]) + "/" + path_name_lst[1])
                    # cv2.VideoWriter()
                elif not os.path.isfile("../images/" + os.path.join(label[key]) + "/" + path_name_lst[1] + "/" + image) == True:  # if directory not exists or file not exists
                    copy(os.path.join(image_dir, image), "../images/" + os.path.join(label[key]) + "/" + path_name_lst[1])
                elif os.path.isfile("../images/" + os.path.join(label[key]) + "/" + path_name_lst[1] + "/" + image) == True: # file exists
                    i = 1
                    while True:
                        new_name = os.path.join("../images/" + os.path.join(label[key]) + "/" + path_name_lst[1] + "/" + str(i) + ".jpeg")
                        if not os.path.exists(new_name):
                            copy(os.path.join(image_dir, image), new_name)
                            print(image + " replaced with " + str(i) + ".jpeg")
                            break
                        i += 1

    #make videos
    fps = 0.5

    for i in os.listdir("../images"):
        for k in os.listdir("../images/" + i):
            frames = []
            for j in os.listdir("../images/" + i + k):
                img = cv2.imread("../images/" + i + "/" + j)
                height, width, channels = img.shape
                size = (height, width)
                frames.append(img)
            out = cv2.VideoWriter("../videos/" + i +"/"+ k + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, frames[0].shape)
            for i in range(len(frames)):
                out.write(frames[i])
            out.release()