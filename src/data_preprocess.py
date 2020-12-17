import pandas as pd
import numpy as np
import cv2
import os
from glob import glob
import json
from shutil import copy

labels = "../all_labels/"

# organise the data into required folders.
if __name__=="__main__":

    labels_lst = []
    image_file_lst = []
    # video_folder_name = []
    # count = 0
    for file in glob(labels + "*.json"):
        with open(file, mode='r') as writer:
            text = writer.read()
            res = json.loads(text)
            # print(file)
    
            frames_directory = res['framesDirectory']
            labels_each_video = res['labels']
            labels_lst.append(labels_each_video)
            # print(frames_directory)
    
            path_name_lst = frames_directory.split("\\")
    
            # second = path_name_lst[1][19:27]
    
            if len(path_name_lst) == 2:  # bag not available in labels
                if path_name_lst[1][19:27] == "_frames_":
                    first = path_name_lst[1][:19]
                    third = path_name_lst[1][27:]
                    second_word = first + third
                    second_word = second_word[:19] + "bag" + second_word[19:]
                elif path_name_lst[1][19:22] == "bag":
                    first_word = path_name_lst[0]
                    second_word = path_name_lst[1]
                else:
                    first_word = path_name_lst[0]
                    second_word = path_name_lst[1]
                    second_word = second_word[:19] + "bag" + second_word[19:]
                # print(second_word)
            # if len(path_name_lst) == 3: # bag not available in labels
            #     first_word = path_name_lst[0]
            #     second_word = path_name_lst[2]
            #     second_word = second_word[:19] + "bag" + second_word[19:]
                # print(second_word)
            elif len(path_name_lst) == 1:
                if path_name_lst[0][22:23] == "-":
                    first = path_name_lst[0][:22]
                    third = path_name_lst[0][23:]
                    x = first + third
                else:
                    x = path_name_lst[0]
                second_word = x + ".avi"
                # print(second_word)
            image_file_lst.append("../" + "new_raw/" + second_word + "/")
    # print(count)
    
    # # print(labels_lst[0])
    # print(len(image_file_lst))
    # # print(video_folder_name[0])
    for image_dir, label in zip(image_file_lst, labels_lst):
        # print(len(image_dir))
        dir = image_dir.split("/")
        # print(dir)
        image_lst = os.listdir(image_dir)
        for image, key in zip(image_lst, label):
            # print(image)
            # print(label[key])
    
            if not os.path.exists("../images/" + os.path.join(label[key])): #  if directory not exists
                os.mkdir("../images/" + os.path.join(label[key])) # create directory
            if not os.path.exists("../images/" + os.path.join(label[key]) + "/" + dir[2]): # if subdirectory not exists
                os.mkdir("../images/" + os.path.join(label[key]) + "/" + dir[2])
                # copy(os.path.join(image_dir, image), "../images/" + os.path.join(label[key]) + "/" + dir[2])
                # cv2.VideoWriter()
            if os.path.isfile("../images/" + os.path.join(label[key]) + "/" + dir[2] + "/" + image) == False:  # if directory exists or file exists
                copy(os.path.join(image_dir, image), "../images/" + os.path.join(label[key]) + "/" + dir[2])
            if os.path.isfile("../images/" + os.path.join(label[key]) + "/" + dir[2] + "/" + image) == True: # file exists
                i = 1
                while True:
                    new_name = os.path.join("../images/" + os.path.join(label[key]) + "/" + dir[2] + "/" + str(i) + ".jpeg")
                    if not os.path.exists(new_name):
                        copy(os.path.join(image_dir, image), new_name)
                        print(image + " replaced with " + str(i) + ".jpeg")
                        break
                    i += 1

