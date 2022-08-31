import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.io as sio
import datetime
import os
import training_testing as tt
import pandas as pd
import usefull_functions as usf




class DataPreparation:
    def __init__(self, x_init, y_init, test_size=0.15, val_size=0.2,option_save=False,normalization=False,slice_padding=False,slice_padding_size=128, seed=123456):
        
        self.seed=seed
        # reshape the data
        self.reshape_data3D(x_init, y_init)
        self.X_max = 'None'
        self.Y_max = 'None'
        
        
        if slice_padding:
            #self.slice_padding_size=slice_padding_size
            self.padding_slice_data(slice_padding_size)

        now = datetime.datetime.now()
        self.dir_name = now.strftime("%d%m%Y")
        self.check_dir()

        # normalization
        if normalization:
            self.data_normalization()
        # train/test split
        self.build_train_test(test_size=test_size, val_size=val_size,option_save=option_save)

    def check_dir(self):

        dir = os.path.curdir + '/' + self.dir_name
        if not os.path.isdir(dir):
            os.makedirs(dir)

    def reshape_data3D(self, x_init, y_init):
        dim_x = x_init.shape
        dim_y = y_init.shape

        x = x_init.reshape(dim_x[0], dim_x[1], dim_x[2], dim_x[3], 1)
        y = y_init.reshape(dim_y[0], dim_y[1], dim_y[2], dim_y[3], 1)
        self.X = np.transpose(x, (3, 0, 1, 2, 4))
        self.Y = np.transpose(y, (3, 0, 1, 2, 4))

    def data_normalization(self):
        self.X_max = np.amax(np.abs(self.X), axis=(1, 2, 3), keepdims=True).max()
        self.X = self.X / self.X_max

        self.Y_max = np.amax(np.abs(self.Y), axis=(1, 2, 3), keepdims=True).max()
        self.Y = self.Y / self.Y_max

    def data_unnormalization(self):
        self.X = self.X * self.X_max
        self.Y = self.Y * self.Y_max

    def build_train_test(self, test_size=0.1,  val_size=0.1,option_save=False):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_size,                                                                               random_state=self.seed)
        
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=val_size,                                                             random_state=self.seed)

        if option_save:
            np.save(self.dir_name + '/X_train.npy', self.X_train)
            np.save(self.dir_name + '/X_val.npy', self.X_val)
            np.save(self.dir_name + '/X_test.npy', self.X_test)
            np.save(self.dir_name + '/Y_train.npy', self.Y_train)
            np.save(self.dir_name + '/Y_test.npy', self.Y_test)
            np.save(self.dir_name + '/Y_val.npy', self.Y_val)
            
    def padding_slice_data(self,padding):
               
        shapeX=self.X.shape
        shapeY=self.Y.shape
        
        X_pad = np.zeros((shapeX[0],shapeX[1],shapeX[2],padding-shapeX[3],shapeX[4])) 
        Y_pad = np.zeros((shapeY[0],shapeY[1],shapeY[2],padding-shapeY[3],shapeY[4]))

        self.X=np.concatenate((self.X,X_pad),axis=3)
        self.Y=np.concatenate((self.Y,Y_pad),axis=3)
        


# ----------------------------------------------------------------

class PlotData:

    def __init__(self, path='./'):
        plt.set_cmap('bone')
        self.path = path



    def dataplot_predict(self, Xpredict, Xinit, slice=18):
        number_plot = Xpredict.shape[0]
        for num in np.arange(number_plot):
            plt.figure()

            plt.subplot(1, 2, 2)
            plt.imshow(Xpredict[num, :, :, slice, 0])
            plt.title('Prediction')

            plt.subplot(1, 2, 1)
            plt.imshow(Xinit[num, :, :, slice, 0])
            plt.title('Initial')

    def dataplot_predict_ground_truth(self, y_predict, x_init, y, selected_slice=18, label='',directory='',i=1):
        number_plot = y_predict.shape[0]
        for num in np.arange(number_plot):
            plt.figure(i)

            plt.ion()
            plt.subplot(1, 3, 2)
            plt.imshow(y_predict[num, :, :, selected_slice, 0])
            plt.title('Prediction')

            plt.subplot(1, 3, 1)
            plt.imshow(x_init[num, :, :, selected_slice, 0])
            plt.title('Initial')

            plt.subplot(1, 3, 3)
            plt.imshow(y[num, :, :, selected_slice, 0])
            plt.title('Ground truth')
            plt.savefig(directory + '/plot_' + label + str(i) + '.png')
            plt.close(i)

    def generatetitle(self, directory_model):
        hyperparam = tt.VisualizeParameters(directory_model)
        lr = hyperparam.learning_rate

        title = 'learning rate = ' + str(lr)
        return title


    def plot_history(self, directory,i,title = 'loss'):

        #title = self.generatetitle(directory)

        history = pd.read_json(directory + "/history.json")
        plt.figure(i)
        plt.plot(history.loss)
        plt.plot(history.val_loss)
        plt.title(title)
        plt.savefig(directory + "/loss_val.png")
        plt.close(i)
        return history




    def predict_y(self, model_name, x):
        evaluate = tt.EvaluateModel(model_name)
        y_predict = evaluate.predict_data(x)
        return y_predict

    def plot_all(self):

        listdir = usf.commonfunctions().list_directory(directory=self.path)
        print(listdir)
        i=0
        lr_list=[]
        loss_list = []
        val_loss_list = []
        for directory in listdir:
            hyperparam = tt.VisualizeParameters(directory)
            lr = hyperparam.learning_rate

            title = 'learning rate = ' + str(lr)
            history = self.plot_history(directory=directory,i=i,title=title)
            lr_list.append(lr)
            loss_list.append(history.loss)
            val_loss_list.append(history.val_loss)

            i=i+1

        plt.figure(1)
        plt.subplot(1,2,1)
        plt.hist(lr_list,bins=30)

        plt.subplot(1,2,2)
        plt.plot(lr_list,loss_list)
        plt.plot(lr_list,val_loss_list)
        plt.legend(['loss','val_loss'])
        plt.savefig(self.path + 'tested_lr.png')
        plt.close(1)








class LoadData:

    def __init__(self, path_data_x, path_data_y, key_x='full_FB', key_y='full_BH'):
        self.path_data_X = path_data_x
        self.path_data_Y = path_data_y

        # if matlab ------------
        self.key_X = key_x
        self.key_Y = key_y

        self.check_type()

    def check_type(self):
        self.type = self.path_data_X[-4:]
        if self.type == '.mat':
            self.loadmatlab()
        elif self.type == '.npy':
            self.loadnumpy()
        else:
            print('Unknowned data type')

    def loadmatlab(self):
        self.Xinit = sio.loadmat(self.path_data_X)[self.key_X]
        self.Yinit = sio.loadmat(self.path_data_Y)[self.key_Y]

    def loadnumpy(self):
        self.Xinit = np.load(self.path_data_X)
        self.Yinit = np.load(self.path_data_Y)


