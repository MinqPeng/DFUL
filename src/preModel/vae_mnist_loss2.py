# train the VAE on mnist dataset
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Dense, Input, Conv2D, Flatten, Lambda, Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from scipy.stats import norm

#================Reparameterization trick===============
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

#==============================================
def plot_results(models,
                 data,
                 batch_size=128):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
    """
    encoder, decoder = models
    x_test, y_test = data

    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(15, 15))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    
    # display a 30x30 2D manifold of digits
    n = 30
    scale = 4
    digit_size = 28
    figure = np.zeros((digit_size*n, digit_size*n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]
  
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
#             z_sample = np.tile(z_sample,1).reshape(1, 2)
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size,digit_size)
            figure[i*digit_size : (i+1)*digit_size, j*digit_size : (j+1)*digit_size] = digit
  
    plt.figure(figsize=(15, 15))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

#==================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE training')
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use MSE loss or Binary Cross Entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()

    # Load mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # 60000, 10000
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    # Normalization
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Network parameters
    input_shape = (image_size, image_size, 1)
    batch_size = 128
    latent_dim = 32
    epochs = 100

    # VAE model = Encoder + Decoder
    # Build Encoder Model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    x = Conv2D(filters=32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x = Conv2D(filters=32, kernel_size=3, activation='relu', strides=1, padding='same')(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x = Conv2D(filters=64, kernel_size=3, activation='relu', strides=1, padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='VAE_Encoder')
    encoder.summary()
    plot_model(encoder, to_file='./image/VAE/mnist_encoder.jpg', show_shapes=True)

    # Build Decoder Model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', strides=1, padding='same')(x)
    x = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', strides=1, padding='same')(x)
    x = Conv2DTranspose(filters=32, kernel_size=3, activation='relu', strides=2, padding='same')(x)
    decoder_outputs = Conv2DTranspose(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(x)

    # Instantiate decoder model
    decoder = Model(latent_inputs, decoder_outputs, name='VAE_Decoder')
    decoder.summary()
    plot_model(decoder, to_file='./image/VAE/mnist_decoder.jpg', show_shapes=True)

    # Instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='VAE_mnist')

    # VAE loss = reconstruction_loss + kl_loss
    if args.mse:
        # mse_loss
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    else:
        # xent_loss
        reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= image_size * image_size
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae, to_file='./image/VAE/mnist.jpg', show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        print('Saving model...')
        # save whole
        encoder.save('./model/VAE/mnist_Encoder.h5')
        # save Model to json
        encoder_json = encoder.to_json()
        with open('./model/VAE/mnist_Encoder.json', 'w') as file:
            file.write(encoder_json)
        # save Model weights
        encoder.save_weights('./model/VAE/mnist_EncoderWeights.h5')
        # save whole
        decoder.save('./model/VAE/mnist_Decoder.h5')
        # save Model to json
        decoder_json = decoder.to_json()
        with open('./model/VAE/mnist_Decoder.json', 'w') as file:
            file.write(decoder_json)
        # save Model weights
        decoder.save_weights('./model/VAE/mnist_DecoderWeights.h5')
        # save whole
        vae.save('./model/VAE/mnist_VAE.h5')
        # save Model to json
        vae_json = vae.to_json()
        with open('./model/VAE/mnist_VAE.json', 'w') as file:
            file.write(vae_json)
        # save Model weights
        vae.save_weights('./model/VAE/mnist_VAEWeights.h5')   
        print('done!')
        
#=====================Visualization=====================
# for latent_dim = 2 
#     models = (encoder, decoder)
#     data = (x_test, y_test) 
#     plot_results(models, data, batch_size=batch_size)

