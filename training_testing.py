#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

#import tf.keras.models
import tensorflow as tf
# import tensorflow.keras
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, Dense, concatenate, Dropout, ZeroPadding3D
from tensorflow.keras.regularizers import l2

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pandas as pd

import data_prep_functions


class Train:
    def __init__(self, model, X_train, Y_train, lr=0.001, epochs=100, batchsize=3, loss='mae', metrics=['mse'],
                 model_name="test"):
        self.optimizer = Adam(learning_rate=lr,
                              beta_1=0.9,
                              beta_2=0.999,
                              epsilon=1E-8,
                              decay=0.0,
                              amsgrad=False)
        self.model = model

        self.filepath_checkpoint = model_name
        self.model_name = model_name
        self.monitor = 'val_loss'
        self.metrics = metrics
        self.loss = loss  # 'mae' # loss

        # compile the model
        self.compile()
        self.model_callbacks = [self.model_checkpoint]
        # fit
        self.epochs = epochs
        self.validation_split = 0.2
        self.batch_size = batchsize

        self.train_model(X_train, Y_train)

    def compile(self):
        self.model_checkpoint = ModelCheckpoint(self.filepath_checkpoint,
                                                monitor=self.monitor,
                                                save_best_only=True)
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

    def train_model(self, x_train, y_train):
        history = self.model.fit(x_train, y_train, batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 validation_split=self.validation_split,
                                 shuffle=True,
                                 verbose=0,
                                 callbacks=self.model_callbacks)

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)

        # save to json:
        hist_json_file = self.model_name + '/history.json'
        if not os.path.isdir(self.model_name):
            os.mkdir(self.model_name)

        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)


class EvaluateModel:
    def __init__(self, model_name):
        self.model = tf.keras.models.load_model(model_name)

    def predict_data(self, x, batch_size=3, generate_array=True):
        if generate_array:
            nb = np.arange(batch_size)
        else:
            nb = batch_size
        y_predict = self.model.predict(x[nb,:,:,:,:])
        return y_predict

    def evaluate_model(self, path_data_x, path_data_y):
        input_data = data_prep_functions.LoadData(path_data_x = path_data_x,path_data_y=path_data_y)
        data_x = input_data.Xinit
        data_y = input_data.Yinit

        eval_model = self.model.evaluate(data_x,data_y,batch_size=3)
        return eval_model



class VisualizeParameters:
    def __init__(self,model_name):
        self.model = keras.models.load_model(model_name)
        self.load_hyper_optimizer()

    def load_hyper_optimizer(self):
        opt = self.model.optimizer.get_config()
        self.learning_rate = opt['learning_rate']




