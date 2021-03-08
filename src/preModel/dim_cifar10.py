import numpy as np
import pandas as pd
# import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import Adam
from keras.datasets import cifar10
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#==================================================
def load_data_from_npz():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data('../../datasets/cifar-10-python.tar.gz')
    # reshape and standardize x arrays
    x_train = x_train.astype("float32") / 255 - 0.5
    x_test = x_test.astype("float32") / 255 - 0.5
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    return x_train, x_test, y_train, y_test

def load_data_from_csv():
    x_train = pd.read_csv('../../csv/base/cifar10/x_train.csv', header=None)
    x_train = x_train.values.astype('float32')
    x_val = pd.read_csv('../../csv/base/cifar10/x_val.csv', header=None)
    x_val = x_val.values.astype('float32')
    return x_train, x_val

#==================================================
def sampling(args):
    z_mean, z_log_var = args
    u = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * u

#==================================================
def shuffling(x):
    idxs = K.arange(0, K.shape(x)[0])
    idxs = tf.random_shuffle(idxs)
    return K.gather(x, idxs)

#====================Visualization======================
def sample_knn(path):
    n = 10
    topn = 10
    figure1 = np.zeros((img_dim*n, img_dim*topn, 3))
    figure2 = np.zeros((img_dim*n, img_dim*topn, 3))
    zs_ = zs / (zs**2).sum(1, keepdims=True)**0.5
    for i in range(n):
        one = np.random.choice(len(x_train))
        idxs = ((zs**2).sum(1) + (zs[one]**2).sum() - 2 * np.dot(zs, zs[one])).argsort()[:topn]
        for j,k in enumerate(idxs):
            digit = x_train[k]
            figure1[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
        idxs = np.dot(zs_, zs_[one]).argsort()[-n:][::-1]
        for j,k in enumerate(idxs):
            digit = x_train[k]
            figure2[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure1 = (figure1 + 1) / 2 * 255
    figure1 = np.clip(figure1, 0, 255)
    figure2 = (figure2 + 1) / 2 * 255
    figure2 = np.clip(figure2, 0, 255)
    imageio.imwrite(path+'_l2.jpg', figure1)
    imageio.imwrite(path+'_cos.jpg', figure2)

#===================Load Data=======================
x_train, x_test, y_train, y_test = load_data_from_npz()
img_dim = x_train.shape[1]

z_dim = 256 
alpha = 0.5
beta = 1.5 
gamma = 0.01 

#===================Build Model=======================
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

for i in range(3):
    x = Conv2D(z_dim // 2**(2-i),
               kernel_size=(3,3),
               padding='SAME')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = MaxPooling2D((2, 2))(x)

feature_map = x 
feature_map_encoder = Model(x_in, x)

for i in range(2):
    x = Conv2D(z_dim,
               kernel_size=(3,3),
               padding='SAME')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = GlobalMaxPooling2D()(x) 

z_mean = Dense(z_dim)(x)
z_log_var = Dense(z_dim)(x) 

encoder = Model(x_in, z_mean)
plot_model(encoder, to_file='./image/DIM/cifar10_encoder.jpg', show_shapes=True)

z_samples = Lambda(sampling)([z_mean, z_log_var])
prior_kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))

z_shuffle = Lambda(shuffling)(z_samples)
z_z_1 = Concatenate()([z_samples, z_samples])
z_z_2 = Concatenate()([z_samples, z_shuffle])

feature_map_shuffle = Lambda(shuffling)(feature_map)
z_samples_repeat = RepeatVector(4 * 4)(z_samples)
z_samples_map = Reshape((4, 4, z_dim))(z_samples_repeat)
z_f_1 = Concatenate()([z_samples_map, feature_map])
z_f_2 = Concatenate()([z_samples_map, feature_map_shuffle])

#====================Global======================
z_in = Input(shape=(z_dim*2,))
z = z_in
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(1, activation='sigmoid')(z)

GlobalDiscriminator = Model(z_in, z)

z_z_1_scores = GlobalDiscriminator(z_z_1)
z_z_2_scores = GlobalDiscriminator(z_z_2)
global_info_loss = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))


#====================Local======================
z_in = Input(shape=(None, None, z_dim*2))
z = z_in
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(z_dim, activation='relu')(z)
z = Dense(1, activation='sigmoid')(z)

LocalDiscriminator = Model(z_in, z)

z_f_1_scores = LocalDiscriminator(z_f_1)
z_f_2_scores = LocalDiscriminator(z_f_2)
local_info_loss = - K.mean(K.log(z_f_1_scores + 1e-6) + K.log(1 - z_f_2_scores + 1e-6))

#==============================================
dim = Model(x_in, [z_z_1_scores, z_z_2_scores, z_f_1_scores, z_f_2_scores])
plot_model(dim, to_file='./image/DIM/cifar10_DIM.jpg', show_shapes=True)
dim.add_loss(alpha * global_info_loss + beta * local_info_loss + gamma * prior_kl_loss)
dim.compile(optimizer=Adam(1e-3))

dim.fit(x_train, epochs=50, batch_size=64)

print('Saving model...')
# save whole
encoder.save('./model/DIM/cifar10_Encoder.h5')
# save Model to json
encoder_json = encoder.to_json()
with open('./model/DIM/cifar10_Encoder.json', 'w') as file:
    file.write(encoder_json)
# save Model weights
encoder.save_weights('./model/DIM/cifar10_EncoderWeights.h5')
# save whole
feature_map_encoder.save('./model/DIM/cifar10_fEncoder.h5')
# save Model to json
feature_map_encoder_json = feature_map_encoder.to_json()
with open('./model/DIM/cifar10_fEncoder.json', 'w') as file:
    file.write(feature_map_encoder_json)
# save Model weights
feature_map_encoder.save_weights('./model/DIM/cifar10_fEncoderWeights.h5')
# save whole
dim.save('./model/DIM/cifar10_DIM.h5')
# save Model to json
dim_json = dim.to_json()
with open('./model/DIM/cifar10_DIM.json', 'w') as file:
    file.write(dim_json)
# save Model weights
dim.save_weights('./model/DIM/cifar10_DIMWeights.h5')   
print('done!')

zs = encoder.predict(x_train, verbose=True)
zs.mean() 
zs.std() 

sample_knn('test')

#==============================================
# create vade embed data 
x_train, x_val = load_data_from_csv()
x_train = x_train.reshape((len(x_train), 32, 32, 3))
x_val = x_val.reshape((len(x_val), 32, 32, 3))
x_train_embed = encoder.predict(x_train, batch_size=1000)
save = pd.DataFrame(x_train_embed)  
save.to_csv('../../csv/embed/cifar10/x_train_embed_dim.csv', index=False, sep=',', header=None) 
x_val_embed = encoder.predict(x_val, batch_size=1000)
save = pd.DataFrame(x_val_embed)  
save.to_csv('../../csv/embed/cifar10/x_val_embed_dim.csv', index=False, sep=',', header=None) 
print('embed data saved!')

