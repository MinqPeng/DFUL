'''
data.py: contains all data generating code for datasets used in the script
'''
import numpy as np
import pandas as pd
from keras.datasets import mnist, cifar10
from keras.models import Model, load_model
from sklearn import metrics

import pairs
import evaluate
import save
# from util import retrieve_name

#=====================================================================
def get_random(x_test, y_test, features):
    randnum = np.random.randint(0, len(x_test))
    np.random.seed(randnum)
    np.random.permutation(x_test)
    np.random.seed(randnum)
    np.random.permutation(y_test)
    np.random.seed(randnum)
    np.random.permutation(features)
    return x_test, y_test, features

#=====================================================================
def get_mnist():
    '''
    Returns the train and evaluate splits of the MNIST digits dataset,
    where x_train and x_test are shaped into the tensorflow image data
    shape and normalized to fit in the range [0, 1]
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape and standardize x arrays
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255
    return x_train, x_test, y_train, y_test

def get_cifar10():
    '''
    Returns the train and evaluate splits of the CIFAR10 digits dataset,
    where x_train and x_test are shaped into the tensorflow image data
    shape and normalized to fit in the range [0, 1]
    '''
    (x_train, y_train), (x_test, y_test) = cifar10.load_data('../datasets/cifar-10-python.tar.gz')
    # reshape and standardize x arrays
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    
#     Y_train = np_utils.to_categorical(y_train, 10)
#     Y_test = np_utils.to_categorical(y_test, 10)
    return x_train, x_test, y_train, y_test

def load_data_from_npz(dname):
    '''
    Returns the original dataset from downloaded package
    '''
    print('=> load data form npz')
    if dname == 'mnist':
        x_train, x_test, y_train, y_test = get_mnist() # x_train(60000, 28,28,1)
    elif dname == 'cifar10':
        x_train, x_test, y_train, y_test = get_cifar10() # x_train(50000,32,32,3)
    else:
        raise ValueError('Dataset provided ({}) is invalid!'.format(dname))
    return x_train, x_test, y_train, y_test


def load_data_from_csv(path, ctype):
    '''
    Returns the datet from previously saved csv file
    '''
    print('=> load data from csv')
    if ctype == 0:
        data = pd.read_csv(path, header=None)
        data = data.values.reshape(-1)
    elif ctype == 1:
        data = pd.read_csv(path, header=None)
        data = data.values.astype('float32')         
    else:
        raise ValueError('csv type must be 0 or 1!')
    return data

#=====================================================================
def split_data(x, y, split, permute=None):
    '''
    Splits arrays x and y, of dimensionality n x d1 and n x d2, into
    k pairs of arrays (x1, y1), (x2, y2), ..., (xk, yk), where both
    arrays in the ith pair is of shape split[i-1]*n x (d1, d2)

    x, y:       two matrices of shape n x d1 and n x d2
    split:      a list of floats of length k (e.g. [a1, a2,..., ak])
                where a, b > 0, a, b < 1, and a + b == 1
    permute:    a list or array of length n that can be used to
                shuffle x and y identically before splitting it

    returns:    a tuple of tuples, where the outer tuple is of length k
                and each of the k inner tuples are of length 3, of
                the format (x_i, y_i, p_i) for the corresponding elements
                from x, y, and the permutation used to shuffle them
                (in the case permute == None, p_i would simply be
                range(split[0]+...+split[i-1], split[0]+...+split[i]),
                i.e. a list of consecutive numbers corresponding to the
                indices of x_i, y_i in x, y respectively)
    '''
    print('=> split data')
    n = len(x)
    if permute is not None:
        if not isinstance(permute, np.ndarray):
            raise ValueError("Provided permute array should be an np.ndarray, not {}!".format(type(permute)))
        if len(permute.shape) != 1:
            raise ValueError("Provided permute array should be of dimension 1, not {}".format(len(permute.shape)))
        if len(permute) != len(x):
            raise ValueError("Provided permute should be the same length as x! (len(permute) = {}, len(x) = {}".format(len(permute), len(x)))
    else:
        permute = np.arange(len(x))

    if np.sum(split) != 1:
        raise ValueError("Split elements must sum to 1!")

    ret_x_y_p = []
    prev_idx = 0
    for s in split:
        idx = prev_idx + np.round(s * n).astype(np.int)
        p_ = permute[prev_idx:idx]
        x_ = x[p_]
        y_ = y[p_]
        prev_idx = idx
        ret_x_y_p.append((x_, y_, p_))

    return tuple(ret_x_y_p)

#=====================================================================
def embed_data(premodel):
    '''
    Embeds x into the coding space using the corresponding
    preModel (specified by dataset).
    '''
    print('=> embed data in '+ premodel)
    if premodel == 'vae':
        x_train = load_data_from_csv('../csv/base/mnist/x_train.csv', 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_val = load_data_from_csv('../csv/base/mnist/x_val.csv', 1)
        x_val = x_val.reshape(-1, 28, 28, 1)
        vae_model = load_model('./preModel/model/VAE/mnist_Encoder.h5', compile=False)
        m_encoder = Model(vae_model.inputs, vae_model.get_layer('z_mean').output)
        m_encoder.summary
        x_train = m_encoder.predict(x_train, batch_size=1000) 
        save.save_data_to_csv(x_train, '../csv/embed/mnist/x_train_embed_vae.csv')
        x_val = m_encoder.predict(x_val, batch_size=1000) 
        save.save_data_to_csv(x_val, '../csv/embed/mnist/x_val_embed_vae.csv')
        del m_encoder
    elif premodel == 'vade':
        x_train = load_data_from_csv('../csv/embed/mnist/x_train_embed_vade.csv', 1)
        x_val = load_data_from_csv('../csv/embed/mnist/x_val_embed_vade.csv', 1)
    elif premodel == 'dim':
        x_train = load_data_from_csv('../csv/embed/cifar10/x_train_embed_dim.csv', 1)
        x_val = load_data_from_csv('../csv/embed/cifar10/x_val_embed_dim.csv', 1)
#     elif premodel == 'amdim':
#         amdim_model = load_model('../model/AMDIM/amdim_Encoder.h5', compile=False)
#         c_encoder = Model(amdim_model.inputs, amdim_model.get_layer('z_mean').output)
#         x_embedded = c_encoder.predict(x, batch_size=1024, verbose=True)
#         print('x_embed:', x_embedded.shape)
#         del c_encoder 
    else:
        raise ValueError("Premodel must be vae, vade, dim or amdim! Please check your type!")
    return x_train, x_val

#=====================================================================
def create_base_dataset(params, data=None):
    '''
    Creates the base data from the original dataset after shuffle and split
    '''
    print('[Create base data]')
    dname = params['dataset']
    # get data if not provided
    if data is None:
        x_train, x_test, y_train, y_test = load_data_from_npz(dname)
    else:
        print("WARNING: Using data provided in arguments. Must be tuple or array of format (x_train, x_test, y_train, y_test)")
        x_train, x_test, y_train, y_test = data      

    if params.get('use_all_data'):
        x_train = np.concatenate((x_train, x_test), axis=0)
        y_train = np.concatenate((y_train, y_test), axis=0)
        
    # split x training, validation, and evaluate subsets
    if 'val_set_fraction' not in params:
        print("NOTE: Validation set required, setting val_set_fraction to 0.1")
        train_val_split = (.9, .1)
    elif params['val_set_fraction'] > 0 and params['val_set_fraction'] <= 1:
        train_val_split = (1 - params['val_set_fraction'], params['val_set_fraction'])
    else:
        raise ValueError("val_set_fraction is invalid! must be in range (0, 1]")

    # shuffle training and validate data separately into themselves and concatenate
    print('=> shuffle and split')
    p = np.random.permutation(len(x_train))
    (x_train, y_train, p_train), (x_val, y_val, p_val) = split_data(x_train, y_train, train_val_split, permute=p)
    
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))    
    x = np.concatenate((x_train, x_val), axis=0)
    label = np.concatenate((y_train, y_val), axis=0)
    
    # save data after shuffle and split
    print('=> save to csv')
    save.save_data_to_csv(x, '../csv/base/'+ dname +'/x.csv')
    save.save_data_to_csv(x_train, '../csv/base/'+ dname +'/x_train.csv')
    save.save_data_to_csv(x_val, '../csv/base/'+ dname +'/x_val.csv')
    save.save_data_to_csv(label, '../csv/base/'+ dname +'/label.csv')
    save.save_data_to_csv(y_train, '../csv/base/'+ dname +'/y_train.csv')
    save.save_data_to_csv(y_val, '../csv/base/'+ dname +'/y_val.csv')
    save.save_data_to_csv(p_train, '../csv/base/'+ dname +'/p_train.csv')
    save.save_data_to_csv(p_val, '../csv/base/'+ dname +'/p_val.csv')
    save.save_data_to_csv(p, '../csv/base/'+ dname +'/p.csv')
    print('=> training sample visualization')
    if dname == 'mnist':
        evaluate.plot_image(x_train, params, path='../image/'+ dname +'/400samples_mnist.png', title='Sample images of MNIST')
    if dname == 'cifar10':
        evaluate.plot_imageRGB(x_train, params, path='../image/'+ dname +'/400samples_cifar10.png', title='Sample images of CIFAR10')
    print('done!')
    
#=====================================================================
def create_initial_pairwise_dataset(params):
    '''
    Creates the initial pairwise dataset in the coding space (or directly from the base data)
    '''
    print('[Create initial pairwise dataset]')
    dname = params['dataset']
    pmname = params['premodel']
    print('=> load base data')
    y_train = load_data_from_csv('../csv/base/'+ dname +'/y_train.csv', 0)
    y_val = load_data_from_csv('../csv/base/'+ dname +'/y_val.csv', 0)
    p_train = load_data_from_csv('../csv/base/'+ dname +'/p_train.csv', 0)
    p_val = load_data_from_csv('../csv/base/'+ dname +'/p_val.csv', 0)
    
    # embed x_data in coding space, if necessary
    if params.get('use_code_space'):
        print('# use coding space')
        x_train, x_val = embed_data(pmname)
    else:
        if dname == 'mnist':
        # just flatten it
            print('# no coding space, just flatten!')
            x_train = x_train.reshape((-1, np.prod(x_train.shape[1:]))) 
            x_val = x_val.reshape((-1, np.prod(x_val.shape[1:]))) 
        else:
            print('# no coding space, no flatten!')
            
    embed_x = np.concatenate((x_train, x_val), axis=0)
    save.save_data_to_csv(embed_x, '../csv/base/'+ dname +'/'+ pmname +'_embed_x.csv')
    print('$ embedData saved:', embed_x.shape)
    
    # get pairwise dataset if necessary
    if params.get('precomputedKNNPath'):
        # if we use precomputed knn, we cannot shuffle the data; instead
        # we pass the permuted index array and the full matrix so that
        # create_pairs_from_unlabeled data can keep track of the indices
        train_path = params.get('precomputedKNNPath', '')
        if params['val_set_fraction'] < 0.09 or params['siam_k'] > 100:
            # if the validation set is very small, the benefit of
            # the precomputation is small, and there will be a high miss
            # rate in the precomputed neighbors (neighbors that are not
            # in the validation set) so we just recomputed neighbors
            p_val = None
            val_path = ''
        else:
            p_val = p_val[:len(x_val)]
            val_path = params.get('precomputedKNNPath', '')
    else:
        # if we do not use precomputed knn, then this does not matter
        p_train = None
        train_path = params.get('precomputedKNNPath', '')
        p_val = None
        val_path = params.get('precomputedKNNPath', '')

    pairs_train, dist_train, label_train = pairs.create_pairs_from_embed_data(
        x1 = x_train,
        y = y_train,
        x2 = x_train,
        p = p_train,
        k = params.get('siam_k'), # positive pair threshold
        tot_pairs = params.get('siamese_tot_pairs'),
        precomputed_knn_path = train_path,
        use_approx = params.get('use_approx', False),
        pre_shuffled = True,
    )
    pairs_val, dist_val, label_val= pairs.create_pairs_from_embed_data(
        x1 = x_val,
        y = y_val,
        x2 = x_train,
        p = p_val,
        k = params.get('siam_k'), # positive pair threshold
        tot_pairs = params.get('siamese_tot_pairs'),
        precomputed_knn_path = val_path,
        use_approx = params.get('use_approx', False),
        pre_shuffled = True,
    )
    
    pdata = (pairs_train, dist_train, label_train, pairs_val, dist_val, label_val)
    
    print('==========='+ pmname +' Result(0)===========', file=open("LOG_create_pairwise_dataset.txt", "a"))
    pairdist = np.concatenate((dist_train, dist_val), axis=0)
    pairtrue = np.concatenate((label_train, label_val), axis=0)
    acc = np.mean(pairdist == pairtrue)
    print('Initial pairwise Accuracy of '+ pmname +': ' + str(np.round(acc, 3)), file=open("LOG_create_pairwise_dataset.txt", "a"))
    # get the confusion matrix
    cm = metrics.confusion_matrix(pairtrue, pairdist)
    print('Confusion Matrix:', file=open("LOG_create_pairwise_dataset.txt", "a"))
    print(cm, file=open("LOG_create_pairwise_dataset.txt", "a"))
    print("Number of pairs: %d" %pairtrue.shape[0], file=open("LOG_create_pairwise_dataset.txt", "a"))
    print('Initial pairwise dataset ready!', file=open("LOG_create_pairwise_dataset.txt", "a"))
    
    print('done!') 
    return pdata

#=====================================================================
def create_feature_pairwise_dataset(params, siamtype):
    '''
    Creates the feature pairwise dataset in the deep feature space from siam model
    '''
    print('[Create deep feature pairwise dataset]')
    dname = params['dataset']
    pmname = params['premodel']
    
    y_train = load_data_from_csv('../csv/base/'+ dname +'/y_train.csv', 0)
    y_val = load_data_from_csv('../csv/base/'+ dname +'/y_val.csv', 0)
    p_train = load_data_from_csv('../csv/base/'+ dname +'/p_train.csv', 0)
    p_val = load_data_from_csv('../csv/base/'+ dname +'/p_val.csv', 0)
    features = load_data_from_csv('../csv/deepf/'+ dname +'/'+ pmname +'_initial_siam'+ siamtype +'.csv', 1)
    f_train = features[:len(p_train)]
    f_val = features[len(p_train):]

    # split x training, validation, and evaluate subsets
    if 'val_set_fraction' not in params:
        print("NOTE: Validation set required, setting val_set_fraction to 0.1")
        train_val_split = (.9, .1)
    elif params['val_set_fraction'] > 0 and params['val_set_fraction'] <= 1:
        train_val_split = (1 - params['val_set_fraction'], params['val_set_fraction'])
    else:
        raise ValueError("val_set_fraction is invalid! must be in range (0, 1]")
    
    if params.get('precomputedKNNPath'):
        # if we use precomputed knn, we cannot shuffle the data; instead
        # we pass the permuted index array and the full matrix so that
        # create_pairs_from_unlabeled data can keep track of the indices
        train_path = params.get('precomputedKNNPath', '')
        if train_val_split[1] < 0.09 or params['siam_k'] > 100:
            # if the validation set is very small, the benefit of
            # the precomputation is small, and there will be a high miss
            # rate in the precomputed neighbors (neighbors that are not
            # in the validation set) so we just recomputed neighbors
            p_val = None
            val_path = ''
        else:
            val_path = params.get('precomputedKNNPath', '')
    else:
        # if we do not use precomputed knn, then this does not matter
        p_train = None
        train_path = params.get('precomputedKNNPath', '')
        p_val = None
        val_path = params.get('precomputedKNNPath', '')

    pairs_train, dist_train, tlabel_train = pairs.create_pairs_from_embed_data(
        x1=f_train,
        y=y_train,
        x2=f_train,
        p=p_train,
        k=params.get('siam_k'),
        tot_pairs=params.get('siamese_tot_pairs'),
        precomputed_knn_path=train_path,
        use_approx=params.get('use_approx', False),
        pre_shuffled=True,
    )
    pairs_val, dist_val, tlabel_val= pairs.create_pairs_from_embed_data(
        x1=f_val,
        y=y_val,
        x2=f_val,
        p=p_val,
        k=params.get('siam_k'),
        tot_pairs=params.get('siamese_tot_pairs'),
        precomputed_knn_path=val_path,
        use_approx=params.get('use_approx', False),
        pre_shuffled=True,
    )

    pdata = (pairs_train, dist_train, tlabel_train, pairs_val, dist_val, tlabel_val)     
    print('==========='+ pmname +' Result===========', file=open("LOG_create_pairwise_dataset.txt", "a"))
    pairdist = np.concatenate((dist_train, dist_val), axis=0)
    pairtrue = np.concatenate((tlabel_train, tlabel_val), axis=0)
    acc = np.mean(pairdist == pairtrue)
    print('Feature pairwise Accuracy of '+ pmname +': ' + str(np.round(acc, 3)), file=open("LOG_create_pairwise_dataset.txt", "a"))
    # get the confusion matrix
    cm = metrics.confusion_matrix(pairtrue, pairdist)
    print('Confusion Matrix:', file=open("LOG_create_pairwise_dataset.txt", "a"))
    print(cm, file=open("LOG_create_pairwise_dataset.txt", "a"))
    print("Number of pairs: %d" %pairtrue.shape[0], file=open("LOG_create_pairwise_dataset.txt", "a"))
    print('Feature pairwise dataset ready!', file=open("LOG_create_pairwise_dataset.txt", "a"))
    
    print('done!') 
    return pdata

#=====================================================================
def create_cluster_pairwise_dataset(params, nround):
    '''
    Creates the cluster pairwise dataset from the former clustering result
    '''
    print('[Create cluster pairwise dataset]')
    dname = params['dataset']
    pmname = params['premodel']
    
    x = load_data_from_csv('../csv/pred/'+ dname +'/'+ pmname +'/'+ str(nround-1) +'_embed_x.csv', 1)
    y = load_data_from_csv('../csv/pred/'+ dname +'/'+ pmname +'/'+ str(nround-1) +'_y.csv', 0)
    labels = pd.read_csv('../csv/pred/'+ dname +'/'+ pmname +'/'+ str(nround-1) +'_label.csv')
    flabel = labels['SC_gk_f'].values
    
    num = len(flabel)
    tnum = int(num*0.9)
    y_train = y[:tnum]
    y_val = y[tnum:]
    x_train = x[:tnum]
    x_val = x[tnum:]
    label_train = flabel[:tnum]
    label_val = flabel[tnum:]

    pairs_train, dist_train, tlabel_train = pairs.create_pairs_from_labeled_data(x_train, label_train, y_train)
    pairs_val, dist_val, tlabel_val = pairs.create_pairs_from_labeled_data(x_val, label_val, y_val)
    
    pdata = (pairs_train, dist_train, tlabel_train, pairs_val, dist_val, tlabel_val)
    print('==========='+ pmname +' Result('+ str(nround) +')===========', file=open("LOG_create_pairwise_dataset.txt", "a"))
    pairdist = np.concatenate((dist_train, dist_val), axis=0)
    pairtrue = np.concatenate((tlabel_train, tlabel_val), axis=0)
    acc = np.mean(pairdist == pairtrue)
    print('Cluster pairwise Accuracy of '+ pmname +' : ' + str(np.round(acc, 3)), file=open("LOG_create_pairwise_dataset.txt", "a"))
    # get the confusion matrix
    cm = metrics.confusion_matrix(pairtrue, pairdist)
    print('Confusion Matrix:', file=open("LOG_create_pairwise_dataset.txt", "a"))
    print(cm, file=open("LOG_create_pairwise_dataset.txt", "a"))
    print("Number of pairs: %d" %pairtrue.shape[0], file=open("LOG_create_pairwise_dataset.txt", "a"))
    print('Cluster pairwise dataset ready!', file=open("LOG_create_pairwise_dataset.txt", "a"))
    
    print('done!')
    return pdata

