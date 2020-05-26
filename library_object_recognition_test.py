# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:42:33 2020

@author: Gaël Miramond & Faustine Michel
"""

import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["coordonates"] = []
  #data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["coordonates"].append(f.read())
      #data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset_test(directory):
  objects_df = load_directory_data(os.path.join(directory, "objects"))
  objects_df["polarity"] = 0
  
  return pd.concat([objects_df]).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets_test(path):

  parent_dir = path
  test_df = load_dataset_test(os.path.join(parent_dir,"test"))
  
  return test_df




def get_predictions(estimator, input_fn):
  return [x for x in estimator.predict(input_fn=input_fn, predict_keys="probabilities")]


def best_classes_and_probabilities(estimator_loaded, predict_test_input_fn):
    predictions=get_predictions(estimator_loaded, predict_test_input_fn)
    predicted_classes=[]
    best_probability=[]
    for i in range (len(predictions)) :
        val_max=max(predictions[i]["probabilities"])
        tempo= np.where(predictions[i]["probabilities"]==val_max)
        tempo_class=tempo[0][0]
        predicted_classes.append(tempo_class)
        best_probability.append(val_max)
    return predicted_classes,best_probability

def ambiguity_reject(tab_predictions,tab_probabilities, threshold, number_of_classes):
    tab_predictions_with_reject=[]
    for i in range(0,len(tab_predictions)):
        if tab_probabilities[i]<threshold:
            tab_predictions_with_reject.append(number_of_classes)
        else:
            tab_predictions_with_reject.append(tab_predictions[i])
            
    
    return tab_predictions_with_reject
    


def  object_recognition_classifier(number_of_classes, path_test_dataset, path_model, path_module, reject, path_predicted_labels):
    
    test_df = download_and_load_datasets_test(path_test_dataset)
    # Prediction on the test set.
    predict_test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        test_df, test_df["polarity"], shuffle=False)
    
    embedded_text_feature_column = hub.text_embedding_column(
        key="coordonates", 
        module_spec=path_module)
    
    loaded_ckpt=tf.train.load_checkpoint(path_model)
    estimator_loaded = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=number_of_classes,
        warm_start_from=path_model)
    
    tab_predictions,tab_probabilities=best_classes_and_probabilities(estimator_loaded, predict_test_input_fn)
    
    
    tab_predictions_with_reject=ambiguity_reject(tab_predictions,tab_probabilities,reject,number_of_classes)
    
    
    trgt_predict_txt=open(path_predicted_labels,'w')
    trgt_predict_txt.writelines(str(tab_predictions_with_reject))
    trgt_predict_txt.close()
    
    return
