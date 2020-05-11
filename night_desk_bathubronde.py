

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:31:31 2020

@author: Gaël Miramond & Faustine Michel 
"""

from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
#%%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
#%%
# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  #data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      #data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  # cube_df = load_directory_data(os.path.join(directory, "cube"))
  # sphere_df = load_directory_data(os.path.join(directory, "sphere"))
  monitor_df = load_directory_data(os.path.join(directory, "monitor"))
  night_df = load_directory_data(os.path.join(directory, "night"))
  desk_df = load_directory_data(os.path.join(directory, "desk"))
  bathub_ronde_df = load_directory_data(os.path.join(directory, "bathub_ronde"))
  # cube_df["polarity"] = 1 #cube sont 1 
  # sphere_df["polarity"] = 0
  # monitor_df["polarity"] = 2
  night_df["polarity"] = 0
  desk_df["polarity"] = 1
  bathub_ronde_df["polarity"]=2
  monitor_df["polarity"]=3
  return pd.concat([ night_df, desk_df,bathub_ronde_df, monitor_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
  #dataset = tf.keras.utils.get_file(
      #fname="aclImdb.tar.gz", 
      #origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      #extract=True)
  #parent_dir = r"C:\Users\ASUS\Anaconda3\envs\tensorflow_cpu\essai\\"
  parent_dir = r"C:\Users\ASUS\Anaconda3\envs\tensorflow_cpu\programmegael\desk_night_bathubronde\demo4"
  train_df = load_dataset(os.path.join(parent_dir,"train"))
  test_df = load_dataset(os.path.join(parent_dir,"test"))


  
  return train_df, test_df
#%%
# Reduce logging output.
logging.set_verbosity(logging.ERROR)

train_df, test_df = download_and_load_datasets()
train_df.head()

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    test_df, test_df["polarity"], shuffle=False)



embedded_text_feature_column = hub.text_embedding_column(
    key="sentence", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

#BUILDING 
#We build a DNN classifier 
estimator = tf.estimator.DNNClassifier( 
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=4,
    optimizer=tf.keras.optimizers.Adagrad(lr=0.01))


#TRAINING 
# Training for 5,000 steps means 640,000 training examples with the default
# batch size. This is roughly equivalent to 25 epochs since the training dataset
# contains 25,000 examples.
estimator.train(input_fn=train_input_fn, steps=800);




#TEST
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))

def get_predictions(estimator, input_fn):
  return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

LABELS = ["night","desk","bathub_ronde","monitor"]

#%%
# Create a confusion matrix on training data.
#cm = tf.math.confusion_matrix(train_df["polarity"], 
                              #get_predictions(estimator, predict_train_input_fn))

# Create a confusion matrix on testing data.
cm = tf.math.confusion_matrix(test_df["polarity"], 
                             get_predictions(estimator, predict_test_input_fn))

# Normalize the confusion matrix so that each row sums to 1.
cm = tf.cast(cm, dtype=tf.float32)
cm = cm / tf.math.reduce_sum(cm, axis=1)[:, np.newaxis]

sns.heatmap(cm, annot=True, xticklabels=LABELS, yticklabels=LABELS);
plt.xlabel("Predicted");
plt.ylabel("True");


