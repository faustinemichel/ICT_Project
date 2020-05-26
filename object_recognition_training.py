# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:11:11 2020

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
import sys
import shutil

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["coordonates"] = []
  
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["coordonates"].append(f.read())
      
  return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset_train(directory):

  motor_df = load_directory_data(os.path.join(directory, "motor"))
  gear_df = load_directory_data(os.path.join(directory, "gear"))
  belt_df = load_directory_data(os.path.join(directory, "belt"))

  motor_df["polarity"] = 0
  gear_df["polarity"] = 1
  belt_df["polarity"] = 2
  
  return pd.concat([motor_df, gear_df, belt_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):

  parent_dir = 'demo5'
  train_df = load_dataset_train(os.path.join(parent_dir,"train"))

  
  return train_df


# Reduce logging output.
logging.set_verbosity(logging.ERROR)

train_df = download_and_load_datasets()
train_df.head()

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], shuffle=False)



embedded_text_feature_column = hub.text_embedding_column(
    key="coordonates", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

#Remove the former classifier
shutil.rmtree('Model_tensor')

#BUILDING 
#We build a DNN classifier 
estimator = tf.estimator.DNNClassifier( 
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=3,
    optimizer=tf.keras.optimizers.Adagrad(lr=0.01), model_dir= os.getcwd()+'/Model_tensor')


#TRAINING 
# Training for 5,000 steps means 640,000 training examples with the default
# batch size. This is roughly equivalent to 25 epochs since the training dataset
# contains 25,000 examples.
estimator.train(input_fn=train_input_fn, steps=500);
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
