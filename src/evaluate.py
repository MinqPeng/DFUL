'''
evaluate.py: contains all evaluate functions for methods
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import metrics
from sklearn.manifold import TSNE
from time import time

from util import get_y_preds
# from scipy.stats import mode
import seaborn as sns
sns.set()

#=====================================================================
# Plot gray images (mnist)
def plot_image(data, params, path, title=None):    
    n_img_per_row = 20
    img_size = params['img_size']
    img = np.zeros(((img_size+2) * n_img_per_row, (img_size+2) * n_img_per_row))
    for i in range(n_img_per_row):
        ix = (img_size+2) * i + 1
        for j in range(n_img_per_row):
            iy = (img_size+2) * j + 1
            img[ix : ix+img_size, iy : iy+img_size] = data[i * n_img_per_row + j].reshape((img_size, img_size))

    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1) 
    plt.clf()

# Plot color mages (cifar10)
def plot_imageRGB(data, params, path, title=None):   
    data = data.reshape((len(data), 32, 32, 3))
    n_img_per_row = 9
    img_size = params['img_size']
    img_chl = params['img_chl']
    img = np.zeros(((img_size+2) * n_img_per_row, (img_size+2) * n_img_per_row, img_chl))
    for i in range(n_img_per_row):
        ix = (img_size+2) * i + 1
        for j in range(n_img_per_row):
            iy = (img_size+2) * j + 1
            # array transpose image : (C, H, W) -> (H, W, C)
            img[ix : ix+img_size, iy : iy+img_size, :] = data[i * n_img_per_row + j] 

    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)  
    plt.clf()  
    
#=====================================================================
# Plot the training loss and accuracy
def plot_model_training(H, dname, nround):
    tabl = len(H.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, tabl), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, tabl), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss on ROUND " + str(nround))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('../image/'+ dname +'/'+ str(nround) +'_ModelTrainingResult.jpg', bbox_inches='tight', pad_inches=0.1)
    plt.clf()

#=====================================================================
# Scale and visualize the embedding vectors
def plot_embedding(X, X_emb, label, params, path, title=None): 
    x_min, x_max = np.min(X_emb, 0), np.max(X_emb, 0)
    X_emb = (X_emb - x_min) / (x_max - x_min)
    image_size = params['img_size'] 
    imgge_chl = params['img_chl']
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X_emb.shape[0]):
        plt.text(X_emb[i, 0], X_emb[i, 1], str(label[i]),
                 color=plt.cm.Set1(float(label[i]) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X_emb.shape[0]):
            dist = np.sum((X_emb[i] - shown_images) ** 2, 1)
            if np.min(dist) < 8e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X_emb[i]]]
            if imgge_chl == 1:
                imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i].reshape(image_size, image_size)[::2,::2], cmap=plt.cm.gray_r), X_emb[i]) 
            else:   
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(X[i].reshape(image_size, image_size, -1)[::2,::2,:], cmap=plt.cm.jet), X_emb[i]) 
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)    
    plt.clf() 

def show_tSNE(X, label, datatype, params, imageX=None, path=None):
    print("Computing t-SNE embedding for "+datatype+"...")
    if path is None:
        path='t-sne-test.eps'
    if imageX is None:
        imageX=X
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(imageX, X_tsne, label, params, path, "t-SNE embedding of the "+ datatype +" (time %.2fs)" %(time() - t0)) 

#=====================================================================
def compute_result(pred, data, true, ncluster):
    result = np.zeros(7, dtype=float)
    # get Silhouette Coefficient
    result[0]= metrics.silhouette_score(data, pred)
    # get Calinski Harabasz score
    result[1] = metrics.calinski_harabasz_score(data, pred)
    # get Davies Bouldin score
    result[2] = metrics.davies_bouldin_score(data, pred)
    # get Adjusted Rand index
    result[3] = metrics.adjusted_rand_score(true, pred)
    # get Fowlkes Mallows score 
    result[4] = metrics.fowlkes_mallows_score(true, pred)
    # get Normalized Mutual Information
    result[5] = metrics.normalized_mutual_info_score(true, pred) # str(np.round(nmi_score, 3)

    # get accuracy
    label = get_y_preds(pred, true, ncluster)
    result[6] = metrics.accuracy_score(true, label)
    return result, label

#=====================================================================
def plot_confusionMatrix1(pred, true, nround, params, labelname=None):
    if labelname is None:
        labelname = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    methodname = ['Kmeans_e', 'Kmeans_f', 'SC_knn_e', 'SC_knn_f', 'SC_gk_e', 'SC_gk_ew', 'SC_gk_f', 'SC_gk_fw']
    for i in range(len(methodname)):
        plt.figure(figsize=(8,6), dpi=100)
        plt.title('Normalized confusion matrix of '+ methodname[i], fontdict = {'fontsize' : 16, 'fontweight':'bold'})
        path='../image/'+ params['dataset'] +'/'+ params['premodel'] +'_'+ str(nround) +'_confusionMatrix1_'+ methodname[i] +'.jpg'
        cm = metrics.confusion_matrix(true, pred[methodname[i]])
        sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=True, xticklabels=labelname, yticklabels=labelname)
        plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
        plt.cla()
    plt.clf()

def plot_confusionMatrix2(pred, true, nround, params, label=None):   
    if label is None:
        label = ['0', '1', '2', '3', '4', '5', '6', '7','8', '9']
    methodname = ['Kmeans_e', 'Kmeans_f', 'SC_knn_e', 'SC_knn_f', 'SC_gk_e', 'SC_gk_ew', 'SC_gk_f', 'SC_gk_fw']
    tick_marks = np.array(range(len(label))) + 0.5
    for i in range(len(methodname)):
        cm = metrics.confusion_matrix(true, pred[methodname[i]])
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ind_array = np.arange(len(label))
        x, y = np.meshgrid(ind_array, ind_array)
        plt.figure(figsize=(9,6), dpi=120)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=9, va='center', ha='center')
        # offset the tick
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='major', linestyle='-')
        plt.gca().tick_params(bottom=False, top=False, left=False, right=False, colors='blue', pad=2)
        plt.gcf().subplots_adjust(bottom=0.15)
        
        plot_confusion_matrix(cm_normalized, label, title='Normalized confusion matrix of '+ methodname[i])
        # show confusion matrix
        plt.savefig('../image/'+ params['dataset'] +'/'+ params['premodel'] +'_'+ str(nround) +'_confusionMatrix2_'+ methodname[i] +'.jpg', bbox_inches='tight', pad_inches=0.1)
        plt.cla()
    plt.clf()
        
def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'fontsize' : 16, 'fontweight':'bold'})
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.grid(False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')