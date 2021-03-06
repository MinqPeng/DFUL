import os
os.environ['KERAS_BACKEND']='theano'

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.datasets import mnist
import scipy.io as scio
import theano 
import theano.tensor as T
import math
import keras.utils
from keras.models import model_from_json
from PIL import Image

#==================================================
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

#=====================================
def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, ind

#==================================================
def load_data_from_npz():
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # 60000, 10000
    # Normalization
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    X = np.concatenate((x_train,x_test))
    Y = np.concatenate((y_train,y_test))
    return X, Y

def load_data_from_csv():
    x_train = pd.read_csv('../../csv/base/mnist/x_train.csv', header=None)
    x_train = x_train.values.astype('float32')
    x_val = pd.read_csv('../../csv/base/mnist/x_val.csv', header=None)
    x_val = x_val.values.astype('float32')
    return x_train, x_val
    
#=====================================================       
def gmm_para_init():
    gmm_weights=scio.loadmat('model/VaDE/mnist_weights_gmm.mat')
    u_init=gmm_weights['u']
    lambda_init=gmm_weights['lambda']
    theta_init=np.squeeze(gmm_weights['theta'])
    
    theta_p=theano.shared(np.asarray(theta_init,dtype=theano.config.floatX),name="pi")
    u_p=theano.shared(np.asarray(u_init,dtype=theano.config.floatX),name="u")
    lambda_p=theano.shared(np.asarray(lambda_init,dtype=theano.config.floatX),name="lambda")
    return theta_p,u_p,lambda_p

#==========================
def generation_init():
    gene_weights=scio.loadmat('model/VaDE/mnist_gene.mat')
    u_gene=gene_weights['u']
    lambda_gene=gene_weights['lambda']
    theta_gene=np.squeeze(gene_weights['theta'])
    gene = model_from_json(open('model/VaDE/mnist_gene.json').read())
    gene.load_weights('model/VaDE/mnist_gene_nn.h5')
    return gene,theta_gene,u_gene,lambda_gene

#================================
def get_gamma(tempz):
    temp_Z=T.transpose(K.repeat(tempz,n_centroid),[0,2,1])
    temp_u_tensor3=T.repeat(u_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_lambda_tensor3=T.repeat(lambda_p.dimshuffle('x',0,1),batch_size,axis=0)
    temp_theta_tensor3=theta_p.dimshuffle('x','x',0)*T.ones((batch_size,latent_dim,n_centroid))
    
    temp_p_c_z=K.exp(K.sum((K.log(temp_theta_tensor3)-0.5*K.log(2*math.pi*temp_lambda_tensor3)-\
                       K.square(temp_Z-temp_u_tensor3)/(2*temp_lambda_tensor3)),axis=1))
    return temp_p_c_z/K.sum(temp_p_c_z,axis=-1,keepdims=True)
#=====================================================

ispretrain = True
batch_size = 100
latent_dim = 10
intermediate_dim = [500,500,2000]
theano.config.floatX='float32'
original_dim = 784
n_centroid = 10 
theta_p, u_p, lambda_p = gmm_para_init()

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim[0], activation='relu')(x)
h = Dense(intermediate_dim[1], activation='relu')(h)
h = Dense(intermediate_dim[2], activation='relu')(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
x_decoded_mean = Dense(original_dim, activation='sigmoid')(h_decoded)

p_c_z = Lambda(get_gamma, output_shape=(n_centroid,))(z_mean)
sample_output = Model(x, z_mean)
p_c_z_output = Model(x, p_c_z)
sample_output.summary  
  
vade = Model(x, x_decoded_mean)
vade.summary
vade.load_weights('model/VaDE/mnist_weights_nn.h5')

X,Y = load_data_from_npz()
accuracy,ind = cluster_acc(np.argmax(p_c_z_output.predict(X,batch_size=batch_size),axis=1),Y)
print ('MNIST dataset VaDE - clustering accuracy: %.2f%%'%(accuracy*100))

# create vade embed data 
x_train, x_val = load_data_from_csv()
x_train_embed = sample_output.predict(x_train, batch_size=1000)
save = pd.DataFrame(x_train_embed)  
save.to_csv('../../csv/embed/mnist/x_train_embed_vade.csv', index=False, sep=',', header=None) 
x_val_embed = sample_output.predict(x_val, batch_size=1000)
save = pd.DataFrame(x_val_embed)  
save.to_csv('../../csv/embed/mnist/x_val_embed_vade.csv', index=False, sep=',', header=None) 
print('embed data seved!')

#=====================================================
gene,g_theta,g_u,g_lambda = generation_init()

def get_posterior(z,u,l,sita):
    z_m=np.repeat(np.transpose(z),n_centroid,1)
    posterior=np.exp(np.sum((np.log(sita)-0.5*np.log(2*math.pi*l)-\
                       np.square(z_m-u)/(2*l)),axis=0))
    return posterior/np.sum(posterior,axis=-1,keepdims=True)

def mnist_gene():
    index=np.asarray(ind)[:,1]
    mnist_nice_png=np.zeros((280,280))
    for i in range(10):
        k=np.where(index==i)[0][0]
        u=g_u[:,k]
        l=g_lambda[:,k]
        sample_n=10
        count=0
        while count<sample_n:
            z_sample=np.random.multivariate_normal(u,np.diag(l),(1,))
            p=get_posterior(z_sample,g_u,g_lambda,g_theta)[k]
            if p>0.999:
                img=gene.predict(z_sample).reshape((28,28))*255.0
                mnist_nice_png[i*28:(i+1)*28,count*28:(count+1)*28]=img
                count+=1
    return np.asarray(mnist_nice_png,dtype=np.uint8)     

digit_image = mnist_gene()
plt.imshow(digit_image,cmap=cm.gray)
plt.show()
Image.fromarray(digit_image).save('mnist_gene_vade.jpg')

