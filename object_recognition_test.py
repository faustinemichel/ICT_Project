# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:42:33 2020

@author: Gaël Miramond & Faustine Michel
"""

from absl import logging

import tensorflow as tf

import tensorflow_hub as hub

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
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
  motor_df = load_directory_data(os.path.join(directory, "motor"))
  gear_df = load_directory_data(os.path.join(directory, "gear"))
  belt_df = load_directory_data(os.path.join(directory, "belt"))

  motor_df["polarity"] = 0
  gear_df["polarity"] = 1
  belt_df["polarity"] = 2
  
  return pd.concat([motor_df, gear_df, belt_df]).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets_test(force_download=False):

  parent_dir = 'demo5'
  test_df = load_dataset_test(os.path.join(parent_dir,"test"))
  
  return test_df


test_df = download_and_load_datasets_test()
# Prediction on the test set.
predict_test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    test_df, test_df["polarity"], shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="coordonates", 
    module_spec=r"C:\Users\ASUS\Desktop\module\\")

loaded_ckpt=tf.train.load_checkpoint(r"C:\Users\ASUS\Anaconda3\envs\tensorflow_cpu\programmegael\recognize\Model_tensor")
estimator_loaded = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=3,
    warm_start_from=r"C:\Users\ASUS\Anaconda3\envs\tensorflow_cpu\programmegael\recognize\Model_tensor")



test_eval_result = estimator_loaded.evaluate(input_fn=predict_test_input_fn)


print("Test set accuracy: {accuracy}".format(**test_eval_result))



def get_predictions(estimator, input_fn):
  return [x for x in estimator.predict(input_fn=input_fn, predict_keys="probabilities")]


#%%
def best_classes_and_probabilities():
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

def ambiguity_reject(tab_predictions,tab_probabilities, threshold):
    tab_predictions_with_reject=[]
    for i in range(0,len(tab_predictions)):
        if tab_probabilities[i]<threshold:
            tab_predictions_with_reject.append(3)
        else:
            tab_predictions_with_reject.append(tab_predictions[i])
            
    
    return tab_predictions_with_reject
    

tab_predictions,tab_probabilities=best_classes_and_probabilities()


tab_predictions_with_reject=ambiguity_reject(tab_predictions,tab_probabilities,0.9)


LABELS = ["motor","gear", "belt"]


# Create a confusion matrix on training data.
#cm = tf.math.confusion_matrix(train_df["polarity"], 
                              #get_predictions(estimator, predict_train_input_fn))
trgt_true_txt=open("true_labels.txt",'w')
trgt_predict_txt=open("predicted_labels.txt",'w')
trgt_true_txt.writelines(str(test_df["polarity"]))
trgt_predict_txt.writelines(str(tab_predictions_with_reject))
trgt_true_txt.close()
trgt_predict_txt.close()

# Create a confusion matrix on testing data.
cm = tf.math.confusion_matrix(test_df["polarity"], tab_predictions_with_reject)

# Normalize the confusion matrix so that each row sums to 1.
cm = tf.cast(cm, dtype=tf.float32)
cm = cm / tf.math.reduce_sum(cm, axis=1)[:, np.newaxis]

sns.heatmap(cm, annot=True, xticklabels=LABELS, yticklabels=LABELS);
plt.xlabel("Predicted");
plt.ylabel("True");
