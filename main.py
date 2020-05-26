# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:31:40 2020

@author: ASUS
"""
import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import library_object_recognition_test as lort

number_of_classes=3
reject=0.7
path_test_dataset='demo6'
path_module = r"C:\Users\ASUS\Desktop\module\\" 
path_model = r"C:\Users\ASUS\Anaconda3\envs\tensorflow_cpu\programmegael\recognize\Model_tensor"
path_predicted_labels = "predicted_labels.txt"


lort.object_recognition_classifier(number_of_classes, path_test_dataset, path_model, path_module, reject, path_predicted_labels)
