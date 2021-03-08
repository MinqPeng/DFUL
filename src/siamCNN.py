'''
siamDense.py: contains network definitions (for siamese network, triplet siamese network)
'''
from __future__ import print_function
from __future__ import division
from keras.models import Model
from keras.layers import Lambda
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D, merge, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import *

import pandas
import matplotlib.pyplot as plt
import pickle
import math
# import numpy as np
import cost
import util
from LSUV import *
from save import saveModel

class AllCNN:
    """AllCNN encapsulates the All-CNN.
    """
    def __init__(self, inputs, is_dropout=True, is_bn=False, seed=22, initializer="glorot_uniform", is_init_fixed=True):
        self.orig_inputs = inputs
        # set up inputs
        self.inputs = {
                'A': inputs['Unlabeled1'],
                'B': inputs['Unlabeled2'],
                }
        self.outputs = dict()
#         self.seed = seed
#         np.random.seed(seed)

        # build the network architecture
        for key in self.inputs:
            if initializer != "LSUV":
                x = Conv2D(96, (3, 3), padding='same', kernel_initializer=initializer)(self.inputs[key])
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(96, (3, 3), padding='same', kernel_initializer=initializer)(x)
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(96, (3, 3), padding='same', strides=2, kernel_initializer=initializer)(x)
                if is_dropout:
                    x = Dropout(0.5)(x)
    
                x = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializer)(x)
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializer)(x)
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(192, (3, 3), padding='same', strides=2, kernel_initializer=initializer)(x)
                if is_dropout:
                    x = Dropout(0.5)(x)
    
                x = Conv2D(192, (3, 3), padding='same', kernel_initializer=initializer)(x)
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(192, (1, 1), padding='valid', kernel_initializer=initializer)(x)
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(10, (1, 1), padding='valid', kernel_initializer=initializer)(x)
    
                x = GlobalAveragePooling2D()(x)

            else:
                x = Conv2D(96, (3, 3), padding='same')(self.inputs[key])
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(96, (3, 3), padding='same')(x)
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(96, (3, 3), padding='same', strides=2)(x)
                if is_dropout:
                    x = Dropout(0.5)(x)
    
                x = Conv2D(192, (3, 3), padding='same')(x)
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(192, 3, 3, padding='same')(x)
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(192, (3, 3), padding='same', strides=2)(x)
                if is_dropout:
                    x = Dropout(0.5)(x)
    
                x = Conv2D(192, (3, 3), padding='same')(x)
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(192, (1, 1), padding='valid')(x)
                if is_bn:
                    x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Conv2D(10, 1, 1, padding='valid')(x)
    
                x = GlobalAveragePooling2D()(x)
                
            self.outputs[key]= x
            
        # add the distance layer    
        self.distance = Lambda(cost.euclidean_distance, output_shape=cost.eucl_dist_output_shape)([self.outputs['A'], self.outputs['B']]) 
        self.net = Model([self.inputs['A'], self.inputs['B']], self.distance)
        

        # set training mode
#         sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#         adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        rmsp = RMSprop(lr=0.001, rho=0.0, epsilon=1e-08, decay=0.001)
        self.net.compile(loss=cost.get_contrastive_loss(m_neg=1, m_pos=0.05), optimizer=rmsp, metrics=['accuracy'])
        print(self.net.summary())
        
    def train(self, pairs_train, dist_train, pairs_val, dist_val, batch_size, epochs, initializer, retrain, dname, pmname, nround): #(self, pairs_train, dist_train, pairs_val, dist_val,
            #lr, drop, patience, num_epochs, batch_size, runtype, dname):
        # path define
        old_weights_path = "oldweights.hdf5"
        new_best_weights_path = "all_cnn_best_weights.hdf5"
        whole_model_path = "all_cnn_whole_model.h5"
        history_path = "all_cnn_history.csv"
    
        accs_epoch_path = "all_cnn_accs_epoch.acc"
        losses_epoch_path = "all_cnn_losses_epoch.loss"
        val_accs_epoch_path = "all_cnn_val_accs_epoch.acc"
        val_losses_epoch_path = "all_cnn_val_losses_epoch.acc"
    
        accs_batch_path = "all_cnn_accs_batch.acc"
        losses_batch_path = "all_cnn_losses_batch.loss"      
        
        acc_figure_path = "acc.png"
        loss_figure_path = "loss.png" 
    
        if not retrain:
            # load pretrainied model
            print("read weights from the pretrained")
            self.net.load_weights(old_weights_path)
        else:
            if initializer == "LSUV":
                # initialize the model using LSUV
                print("retrain the model")
                # training_data_shuffled, training_labels_oh_shuffled = shuffle(X_train, Y_train)
                # batch_xs_init = training_data_shuffled[0:batch_size]
    
                for x_batch, y_batch in util.traingen_flow_for_two_inputs(pairs_train[:,0], pairs_train[:,1], dist_train, batch_size): # make use of image processing utility provided by ImageDataGenerator
                    LSUV_init(self.net, x_batch['pairA'])
                    break
    
        print("start training")
    
        # initialize the callbacks
    
        # save the best model after every epoch
        checkpoint = ModelCheckpoint(new_best_weights_path, monitor='val_acc', verbose=1, save_best_only=True, 
                                     save_weights_only=False, mode='max')
    
        # # print the batch number every batch
        # batch_print_callback = LambdaCallback(on_batch_begin=lambda batch, logs: print(batch))
    
        # learning schedule callback
        lrate = LearningRateScheduler(step_decay)
    
    
        lossAcc = LossAccEveryBatch()
        callbacks_list = [checkpoint, lossAcc]
    
        # fit the model on the batches generated by datagen.flow()
        # it is real-time data augmentation
        history_callback = self.net.fit_generator(util.traingen_flow_for_two_inputs(pairs_train[:,0], pairs_train[:,1], dist_train, batch_size),
                                                steps_per_epoch=int(len(pairs_train) / batch_size),
                                                epochs=epochs, validation_data=util.testgen_flow_for_two_inputs(pairs_val[:,0], pairs_val[:,1], dist_val, batch_size), 
                                                callbacks=callbacks_list, verbose=1, validation_steps=int(len(pairs_train) / batch_size))
    
        pandas.DataFrame(history_callback.history).to_csv(history_path)
        self.net.save(whole_model_path)
    
        # get the stats and dump them for each epoch
        accs_epoch = history_callback.history['acc']
        with open(accs_epoch_path, "w") as fp:  # pickling
            pickle.dump(accs_epoch, fp)
    
        val_accs_epoch = history_callback.history['val_acc']
        with open(val_accs_epoch_path, "w") as fp:  # pickling
            pickle.dump(val_accs_epoch, fp)
    
        losses_epoch = history_callback.history['loss']
        with open(losses_epoch_path, "w") as fp:  # pickling
            pickle.dump(losses_epoch, fp)
    
        val_losses_epoch = history_callback.history['val_loss']
        with open(val_losses_epoch_path, "w") as fp:  # pickling
            pickle.dump(val_losses_epoch, fp)
    
        # get the stats and dump them for each match
        accs_batch = lossAcc.accs_batch
        with open(accs_batch_path, "w") as fp:  # pickling
            pickle.dump(accs_batch, fp)
    
        losses_batch = lossAcc.losses_batch
        with open(losses_batch_path, "w") as fp:  # pickling
            pickle.dump(losses_batch, fp)
    
        # summarize history for accuracy
        plt.plot(history_callback.history['acc'])
        plt.plot(history_callback.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(acc_figure_path, bbox_inches='tight', pad_inches=0.1)
            
        # summarize history for loss
        plt.plot(history_callback.history['loss'])
        plt.plot(history_callback.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(loss_figure_path, bbox_inches='tight', pad_inches=0.1)
            
        return history_callback
    

    def test(self, pairs_val, dist_val, batch_size, epochs):
        old_weights_path = "oldweights.hdf5"
        print("read weights from the pretrained: ", old_weights_path)
        self.net.load_weights(old_weights_path)
    
        util.testgen_flow_for_two_inputs(pairs_val[:,0], pairs_val[:,1], dist_val, batch_size)
        # loss, acc = self.net.evaluate_generator(util.testgen_flow_for_two_inputs(pairs_val[:,0], pairs_val[:,1], dist_val, batch_size),
        #                                        steps_per_epoch=int(len(pairs_val) / batch_size),
        #                                        epochs=epochs, verbose=1)
    
        loss, acc = self.net.evaluate_generator(util.testgen_flow_for_two_inputs(pairs_val[:,0], pairs_val[:,1], dist_val, batch_size),
                                                steps = int(len(pairs_val) / batch_size))
        print("loss: ", loss)
        print("acc: ", acc)
                  
        
class LossAccEveryBatch(Callback):
    """Callback class for saving intermediate acc and loss of each batch
    """
    def on_train_begin(self, logs={}):
        self.losses_batch = []
        self.accs_batch = []

    def on_batch_end(self, batch, logs={}):
        self.losses_batch.append(logs.get('loss'))
        self.accs_batch.append(logs.get('acc'))

def step_decay(epoch):
    """Learning rate scheduler
    """
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

