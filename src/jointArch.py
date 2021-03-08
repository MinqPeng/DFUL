'''
jointArchitech.py: contains run function for joint architecture
'''
import os
# from importlib.resources import path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import numpy as np
import tensorflow.compat.v1 as tf
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input
import math

import cost
import siamDense
import siamCNN
import evaluate
from util import LearningHandler,train_gen
from save import save_data_to_csv, saveModel
from data import load_data_from_csv

def run_siamDense(pdata, params, nround):    
    print('# network type: Dense')
    print('# pairs round: ', nround)
    #
    # SET UP INPUTS
    #
    dname = params['dataset']
    pmname = params['premodel']
    pairs_train, dist_train, label_train, pairs_val, dist_val, label_val = pdata
    
    # SiameseNet has two inputs -- they are defined here
    inputs = {
            'Unlabeled1': Input(shape=(params['input_dim'],),name='pairA'),
            'Unlabeled2': Input(shape=(params['input_dim'],),name='pairB'),
    }
    
    #
    # DEFINE AND TRAIN SIAMESE NET
    #
    if nround == 0:
        dense_model = siamDense.FullDense(inputs, params['arch'], params.get('siam_reg')) 
        print('=> training...')    
        history = dense_model.train(pairs_train, dist_train, pairs_val, dist_val,
                    params['siam_lr'], params['siam_drop'], params['siam_patience'],
                    params['siam_ne'], params['batch_size'])
        print('=> training visualization...')
        evaluate.plot_model_training(history, dname, nround)
        saveModel(dense_model.net, '../model/siamese/'+ dname +'/'+ pmname +'_'+ str(nround) +'_siamdense')
        
    else:  
        dense_model = load_model('../model/siamese/'+ dname +'/'+ pmname +'_'+ str(nround-1) +'_siamdense.h5', custom_objects={'contrastive_loss': cost.get_contrastive_loss(m_neg=1, m_pos=0.05)})
        print('=> retraining...')   
        lh = LearningHandler(
                lr=params['siam_lr'] * math.pow(0.1 , nround),
                drop=0.5,
                lr_tensor=dense_model.optimizer.lr,
                patience=params['siam_patience'])  
        history = dense_model.fit_generator(train_gen(pairs_train, dist_train, 1000), epochs=100, validation_data=([pairs_val[:, 0], pairs_val[:, 1]], dist_val), 
                                              steps_per_epoch=int(len(pairs_train) / 1000), callbacks=[lh])
        print('=> training visualization...')
        evaluate.plot_model_training(history, dname, nround)
        saveModel(dense_model, '../model/siamese/'+ dname +'/'+ pmname +'_'+ str(nround) +'_siamdense')
    K.clear_session()
    print('training accomplished!')     
    
def run_siamCNN(pdata, params, nround):     
    print('# network type: CNN')
    print('# pairs round: ', nround)
    #
    # SET UP INPUTS
    #
    dname = params['dataset']
    pmname = params['premodel']
    pairs_train, dist_train, label_train, pairs_val, dist_val, label_val = pdata
    
#   SiameseNet has two inputs -- they are defined here
    inputs = {
            'Unlabeled1': Input(shape=(params['input_dim'],),name='pairA'),
            'Unlabeled2': Input(shape=(params['input_dim'],),name='pairB'),
    }
    
    #
    # DEFINE AND TRAIN SIAMESE NET
    #
    if nround == 0:     
        # environment configuration
#         K.set_image_dim_ordering('tf')
        # matplotlib.use('Agg') # for server using plt
           
        # initialize the model
        cnn_model = siamCNN.AllCNN(inputs, params['is_dropout'], params['is_bn'])
        print('=> training...')  
        history = cnn_model.train(pairs_train, dist_train, pairs_val, dist_val, params['batch_size'], params['siam_ne'], 
                                   params['initializer'], params['retrain'], dname, pmname, nround)           
        print('=> training visualization...')
        evaluate.plot_model_training(history, dname, nround)
        saveModel(cnn_model.net, '../csv/deepf/'+ dname +'/'+ pmname +'_'+ str(nround) +'_siamcnn')
        
    else:       
        cnn_model = load_model('../model/siamese/'+ dname +'/'+ pmname +'_'+ str(nround-1) +'_siamcnn.h5', custom_objects={'contrastive_loss': cost.get_contrastive_loss(m_neg=1, m_pos=0.05)})    
        print('=> retraining...')    
        lh = LearningHandler(
                lr=params['siam_lr'],
                drop=params['siam_drop'],
                lr_tensor=cnn_model.optimizer.lr,
                patience=params['siam_patience'])
        history = cnn_model.fit_generator(train_gen(pairs_train, dist_train, params['batch_size']), epochs=params['siam_ne'], validation_data=([pairs_val[:, 0], pairs_val[:, 1]], dist_val), 
                                              steps_per_epoch=int(len(pairs_train) / params['batch_size']), callbacks=[lh])
        print('=> training visualization...')
        evaluate.plot_model_training(history, dname, nround)
        saveModel(cnn_model, '../csv/deepf/'+ dname +'/'+ pmname +'_'+ str(nround) +'_siamcnn')
    K.clear_session()    
    print('training accomplished!')        

def deep_feature_gen(params, siamtype, nround, tsize):
    print('=> generate deep feature from trained siam model')
    dname = params['dataset']
    pmname = params['premodel']
    
    print('Load data and model')
    x = load_data_from_csv('../csv/base/'+ dname +'/x.csv', 1)
    label = load_data_from_csv('../csv/base/'+ dname +'/label.csv', 0)
    embed_x = load_data_from_csv('../csv/base/'+ dname +'/'+ pmname +'_embed_x.csv', 1)
    siam_model = load_model('../model/siamese/'+ dname +'/'+ pmname +'_'+ str(nround) +'_siam'+ siamtype +'.h5', custom_objects={'contrastive_loss': cost.get_contrastive_loss(m_neg=1, m_pos=0.05)})
    f_model = Model(siam_model.inputs, siam_model.layers[5].get_output_at(0))
    print('Predict for deep feature')
    features = f_model.predict([embed_x,embed_x], batch_size=1000)
    features = features.astype(np.float32)
    save_data_to_csv(features, '../csv/deepf/'+ dname +'/'+ pmname +'_'+ str(nround) +'_siam'+ siamtype +'.csv')    
    print('$ deepFeature saved:', features.shape)
    
    # Show data distribution of different types
    print('=> data visualization...')
    if nround == 0:  
        evaluate.show_tSNE(x[:tsize], label[:tsize], 'baseData', params, path='../image/'+ dname +'/'+ pmname +'_baseData_'+ siamtype +'.png')
        evaluate.show_tSNE(embed_x[:tsize], label[:tsize], 'embedData', params, imageX=x, path='../image/'+ dname +'/'+ pmname +'_embedData_'+ siamtype +'.png')
    evaluate.show_tSNE(features[:tsize], label[:tsize], 'deepFeature', params, imageX=x, path='../image/'+ dname +'/'+ pmname +'_'+ str(nround) +'_deepFeature_'+ siamtype +'.png')
    tf.reset_default_graph()
    print('done!')
    return features
