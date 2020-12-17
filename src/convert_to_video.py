import pandas as pd
import numpy as np
import cv2
import os
from glob import glob
import json
from shutil import copy

labels = "../all_labels/"

# convert the images to the videos.

if __name__=="__main__":

    for i in os.listdir("../images/"):
        os.mkdir("../videos/uncertain" )
        for k in os.listdir(os.path.join("../images/uncertain")):
            images = [img for img in os.listdir("../images/uncertain" + "/" + k) if img.endswith(".jpeg")]

            frame = cv2.imread(os.path.join("../images/uncertain" + "/" + k, images[0]))
            height, width, layers = frame.shape

            video = cv2.VideoWriter("../videos/uncertain" +"/"+ k + ".mp4", cv2.VideoWriter_fourcc(*'MP4V'), 10, (width, height))

            for image in images:
                video.write(cv2.imread(os.path.join("../images/uncertain" + "/" + k, image)))

            # cv2.destroyAllWindows()
            video.release()