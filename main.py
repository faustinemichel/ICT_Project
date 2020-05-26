"""
Created on Fri May 22 10:02:24 2020
@author: Faustine MICHEL & Gael MIRAMOND
ICT-UNIT040
Main Program 
"""

#Import all the usefull libraries 
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

#Libraries made for the project 
import library_for_classification as lfc
import library_for_pre_post_processing as lfpp

# Global variables for file paths
path_obj_file="obj_file\obj.obj"
path_txt_files=r"txt_files\test\objects\\"
path_object_test = 'txt_files'
path_predicted_labels=r"predictions\predicted_labels.txt"
path_gameObject_names ="gameObjects_in_scene\list_of_gameObjects.txt" 
path_module = r"module\\"
path_model = 'Model_tensor'

# Variables for the classifier
number_of_classes=3
reject=0.7

# -----------------------------Program begins ----------------------------#                
#from the .obj file, as many text files are created as the number of GameObject in the unity scene
lfpp.separate_text(path_obj_file,path_txt_files)

#List all the .txt file in alphabetical order
lfpp.list_files(path_txt_files,path_gameObject_names)

#Classification of the test set by the classifier
lfc.object_recognition_classifier(number_of_classes, path_object_test, path_model, path_module, reject, path_predicted_labels)
#%%
#According to the results 
lfpp.results_processing(path_gameObject_names, path_predicted_labels)