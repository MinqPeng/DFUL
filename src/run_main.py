import sys, os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'

# ADD Directories in src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
from collections import defaultdict
import data
import jointArch
import clustering

# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='0,1,2,3')
parser.add_argument('--dset', type=str, help='dataset to use', default='mnist')
args = parser.parse_args()

# SELECT GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {                 # dataset:  mnist / cifar10
        'val_set_fraction': 0.1,            # fraction of training set to use as validation
        'precomputedKNNPath': '',           # path for precomputed nearest neighbors (with indices and saved as a pickle or h5py file)
        }
params.update(general_params)

# SET DATASET SPECIFIC HYPERPARAMETERS
if args.dset == 'mnist':
    mnist_params = {
        'dataset': 'mnist', 
        'img_size': 28,
        'img_chl': 1,
        'n_clusters': 10,                   # number of clusters in data
        'use_code_space': True,             # enable / disable code space embedding
        'premodel': 'vae',                  # PreModel used to preprocess data
        'affinity': '4Dense',
        'input_dim': 32,
        #####################
        'n_nbrs': 3,                        # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
        'scale_nbr': 2,                     # neighbor used to determine scale of gaussian graph Laplacian; calculated by
                                            # taking median distance of the (scale_nbr)th neighbor, over a set of size batch_size
                                            # sampled from the datset
        'siam_k': 2,                        # threshold where, for all k <= siam_k closest neighbors to x_i, (x_i, k) is considered
                                            # a 'positive' pair by siamese net
        'siam_ne': 100,                     # number of training epochs for siamese net
        'siam_lr': 1e-3,                    # initial learning rate for siamese net
        'siam_patience': 10,                # early stopping patience for siamese net
        'siam_drop': 0.1,                   # learning rate scheduler decay for siamese net
        'batch_size': 128,                  # batch size for siamese net 
        'siam_reg': None,                   # regularization parameter for siamese net
        'siamese_tot_pairs': 600000,        # total number of pairs for siamese net
        'arch': [                           # network architecture. if different architectures are desired for siamese net and
                                            #   spectral net, 'siam_arch' and 'spec_arch' keys can be used
            {'type': 'relu', 'size': 1024},
            {'type': 'relu', 'size': 1024},
            {'type': 'relu', 'size': 512},
            {'type': 'relu', 'size': 10},
            ],
        'use_approx': False,                # enable / disable approximate nearest neighbors
        'use_all_data': True,               # enable to use all data for training (no evaluate set)
        }
    params.update(mnist_params)
    
elif args.dset == 'cifar10':
    cifar10_params = {
        'dataset': 'cifar10', 
        'img_size': 32,
        'img_chl':3,
        'n_clusters': 10,
        'use_code_space': True,
        'premodel': 'dim',      
        'affinity': 'dense',
        'input_dim': 64,
        #####################
        'initializer': "he_uniform",
        'retrain': True, # whether to train from the beginning or read weights from the pretrained model
#         is_training = False, #whether to train or test
        'is_bn': False,
        'is_dropout': False,
        
        'n_nbrs': 3,                        # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
        'scale_nbr': 2,
        'siam_k': 2,
        'siam_ne': 400,
        'siam_lr': 1e-4,
        'siam_patience': 10,
        'siam_drop': 0.3,
        'batch_size': 200,                  # batch size for siamese net 
        'siamese_tot_pairs': 600000,
        'arch': [                           # network architecture. if different architectures are desired for siamese net and
                                            #   spectral net, 'siam_arch' and 'spec_arch' keys can be used
            {'type': 'relu', 'size': 2048},
            {'type': 'BatchNormalization'},
            {'type': 'relu', 'size': 1024},
            {'type': 'BatchNormalization'},
            {'type': 'relu', 'size': 512},
            {'type': 'BatchNormalization'},
            {'type': 'relu', 'size': 64},
            {'type': 'BatchNormalization'},
            ],
        'use_approx': False,
        'use_all_data': True,
        }
    params.update(cifar10_params)

# RUN EXPERIMENT
# print('===[ STEP 1: BASE DATA ]===')
# data.create_base_dataset(params)
 
# print('### LEVEL I : initial ###')
# print('===[ STEP 2: PAIRWISE DATASET ]===') 
# pdata = data.create_initial_pairwise_dataset(params)
# print('===[ STEP 3: BUILD AND TRAIN SIAMESE NETWORK ]===')
# jointArch.run_siamDense(pdata, params, 0)
print('===[ STEP 4: GENERATE DEEP FEATURE ]===')
df = jointArch.deep_feature_gen(params, 'dense', 0, 3000)
print('===[ STEP 5: CLUSTERING AND EVALUATION ]===')   
clustering.run_clustering(df, params, 0, 70)
  
# print('### LEVEL II : cluster - ROUND1 ###')
# print('===[ STEP 2: PAIRWISE DATASET ]===') 
# pdata = data.create_cluster_pairwise_dataset(params, 1)
# print('===[ STEP 3: BUILD AND TRAIN SIAMESE NETWORK ]===')
# jointArch.run_siamDense(pdata, params, 1)
# print('===[ STEP 4: GENERATE DEEP FEATURE ]===')
# df = jointArch.deep_feature_gen(params, 'dense', 1, 3000)
# print('===[ STEP 5: CLUSTERING AND EVALUATION ]===')  
# clustering.run_clustering(df, params, 1, 10)
  
# print('### LEVEL III : cluster - ROUND2 ###')
# print('===[ STEP 2: PAIRWISE DATASET ]===') 
# pdata = data.create_cluster_pairwise_dataset(params, 2)
# print('===[ STEP 3: BUILD AND TRAIN SIAMESE NETWORK ]===')
# jointArch.run_siamDense(pdata, params, 2)
# print('===[ STEP 4: GENERATE DEEP FEATURE ]===')
# df = jointArch.deep_feature_gen(params, 'dense', 2, 3000)
# print('===[ STEP 5: CLUSTERING AND EVALUATION ]===')  
# clustering.run_clustering(df, params, 2, 10)

# print('### LEVEL III : cluster - ROUND3 ###')
# print('===[ STEP 2: PAIRWISE DATASET ]===') 
# pdata = data.create_cluster_pairwise_dataset(params, 3)
# print('===[ STEP 3: BUILD AND TRAIN SIAMESE NETWORK ]===')
# jointArch.run_siamDense(pdata, params, 3)
# print('===[ STEP 4: GENERATE DEEP FEATURE ]===')
# df = jointArch.deep_feature_gen(params, 'dense', 3, 2000)
# print('===[ STEP 5: CLUSTERING AND EVALUATION ]===')  
# clustering.run_clustering(df, params, 3, 20000)

print('DONE!')

