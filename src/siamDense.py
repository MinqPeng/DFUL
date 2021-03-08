'''
siamNet.py: contains network definitions (for siamese network, triplet siamese network)
'''
from keras.models import Model
from keras.layers import Lambda
import cost
from layers import stack_layers
from util import LearningHandler, make_layer_list, train_gen

class FullDense:
    def __init__(self, inputs, arch, siam_reg):
        self.orig_inputs = inputs
        # set up inputs
        self.inputs = {
                'A': inputs['Unlabeled1'],
                'B': inputs['Unlabeled2'],
                }

        # generate layers
        self.layers = []
        self.layers += make_layer_list(arch, 'siamese', siam_reg)
        # create the siamese net
        self.outputs = stack_layers(self.inputs, self.layers)
        # add the distance layer
        self.distance = Lambda(cost.euclidean_distance, output_shape=cost.eucl_dist_output_shape)([self.outputs['A'], self.outputs['B']])
        #create the distance model for training
        self.net = Model([self.inputs['A'], self.inputs['B']], self.distance)
        # compile the siamese network
        self.net.compile(loss=cost.get_contrastive_loss(m_neg=1, m_pos=0.05), optimizer='rmsprop')

    def train(self, pairs_train, dist_train, pairs_val, dist_val,
            lr, drop, patience, num_epochs, batch_size):
        # create handler for early stopping and learning rate scheduling
        self.lh = LearningHandler(
                lr=lr,
                drop=drop,
                lr_tensor=self.net.optimizer.lr,
                patience=patience)
        # initialize the training generator
        train_gen_ = train_gen(pairs_train, dist_train, batch_size)
        # format the validation data for keras
        validation_data = ([pairs_val[:, 0], pairs_val[:, 1]], dist_val)
        # compute the steps per epoch
        steps_per_epoch = int(len(pairs_train) / batch_size)
        # train the network
        hist = self.net.fit_generator(train_gen_, epochs=num_epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch, callbacks=[self.lh])     
        return hist
    
    
