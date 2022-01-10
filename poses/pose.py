# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 12:01:12 2021

@author: IDU
"""
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

path = "C:/Users/IDU/Desktop/dataset"
dir_list = os.listdir(path)

poses =[]
header =  []

doc = []
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
    for idx, file in enumerate(dir_list):
       
        path = "C:/Users/IDU/Desktop/dataset/"+ file
        try:
            image = cv2.imread(path)
        
            image_height, image_width, _ = image.shape
        except:
            print("error")
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
              continue
        
        
        line = []
        for index, each in enumerate(mp_pose.PoseLandmark):
           
            if idx ==0:
                 
                words = str(each).split(".")
                if index == 0:
                    header.append("productid")
                header.append(words[1]+".x")
                header.append(words[1]+".y")  
           
            if index ==0 :
                line.append(str(file))
            line.append(results.pose_landmarks.landmark[each].x * image_width)
            line.append(results.pose_landmarks.landmark[each].y * image_height)
        
        
        doc.append(line)
        
    
    df = pd.DataFrame(doc, columns=header)
    print(df.head())
    df.to_csv('C:/Users/IDU/Desktop/poses_body.csv')  