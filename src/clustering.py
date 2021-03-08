'''
clustering.py: contains all clustering algorithms and result evaluation 
'''
from keras import backend as K
from sklearn.cluster import KMeans, SpectralClustering
from collections import defaultdict

import cost 
import evaluate
from save import saveResult, saveLabel, saveAffinity
from util import get_scale
from data import get_random, load_data_from_csv

def run_clustering_in_subset(sub_x, sub_y, sub_f, params, nround, Pred, i):
    #
    #  RUN GRAPH-BASED ALGORITHMS AND EVALUATION
    #
    print('=> clustering in subset('+ str(i) +')') 
    AffinityW = {}    
    Result = {}
    
    print('Constructing affinity matrix W...') 
    # calculate scale, sigma value for affinity matrix
    scale_e = get_scale(sub_x, params['scale_nbr'])
    scale_f = get_scale(sub_f, params['scale_nbr'])
#     scale_e = get_scale(sub_x, params['batch_size'], params['scale_nbr'])
#     scale_f = get_scale(sub_f, params['batch_size'], params['scale_nbr'])
    print('scale_e:', scale_e)
    print('scale_f:', scale_f)
 
    # create affinity matrix
#     AffinityW['embed'] = cost.full_affinity(sub_x, scale=scale_e)
#     AffinityW['deepf'] = cost.full_affinity(sub_f, scale=scale_f)
    AffinityW['embed_f'] = K.eval(cost.full_affinity(K.variable(sub_x), scale=scale_e))
    AffinityW['deepf_f'] = K.eval(cost.full_affinity(K.variable(sub_f), scale=scale_f))
#     AffinityW['embed_k'] = K.eval(cost.knn_affinity(K.variable(sub_x), scale=scale_e, n_nbrs=params['n_nbrs']))
#     AffinityW['deepf_k'] = K.eval(cost.knn_affinity(K.variable(sub_f), scale=scale_f, n_nbrs=params['n_nbrs']))
    print('Affinity Matrix ready!' )
     
    #
    # UNSUPERVISED LEARNING EVALUATION
    #
    print('Running clustering algorithms...') 
    
    # basedline - K-means
    print('# K-means')
    means_k = KMeans(n_clusters=params['n_clusters'])  
    
    print('$ embed_x')
    pred_ke = means_k.fit_predict(sub_x)
    tresult, tpred = evaluate.compute_result(pred=pred_ke, data=sub_x, true=sub_y, ncluster=params['n_clusters'])
    Result['Kmeans_e'] = tresult
    Pred['Kmeans_e'].extend(tpred)
    
    print('$ deepf')
    pred_kf = means_k.fit_predict(sub_f)
    tresult, tpred = evaluate.compute_result(pred=pred_kf, data=sub_f, true=sub_y, ncluster=params['n_clusters'])
    Result['Kmeans_f'] = tresult
    Pred['Kmeans_f'].extend(tpred)
    
    # Spectral Clustering
    print('# Spectral Clustering (knn)')
    spectral_k = SpectralClustering(affinity='nearest_neighbors', n_clusters=params['n_clusters'], eigen_solver='arpack')
    
    print('$ embed_x')
    pred_sc_ke = spectral_k.fit_predict(sub_x)
    tresult, tpred = evaluate.compute_result(pred=pred_sc_ke, data=sub_x, true=sub_y, ncluster=params['n_clusters'])
    Result['SC_knn_e'] = tresult
    Pred['SC_knn_e'].extend(tpred)
    #spectral_k.fit(sub_x)
    #AffinityW['SC_knn_e'] = spectral_k.affinity_matrix_.todense()
    
    print('$ deepf')
    pred_sc_kf = spectral_k.fit_predict(sub_f)
    tresult, tpred = evaluate.compute_result(pred=pred_sc_kf, data=sub_f, true=sub_y, ncluster=params['n_clusters'])
    Result['SC_knn_f'] = tresult
    Pred['SC_knn_f'].extend(tpred)
    #spectral_k.fit(sub_f)
    #AffinityW['SC_knn_f'] = spectral_k.affinity_matrix_.todense()
    
#     # precomputed Affinity Matrix
#     spectral_kw = SpectralClustering(affinity='precomputed', n_clusters=params['n_clusters'], eigen_solver='arpack')
#     print('$ embed_x(w)')
#     
#     pred_sc_kew = spectral_kw.fit_predict(AffinityW['embed_k'])
#     tresult, tpred = evaluate.compute_result(pred=pred_sc_kew, data=sub_x, true=sub_y, ncluster=params['n_clusters'])
#     Result['SC_knn_ew'] = tresult
#     Pred['SC_knn_ew'].extend(tpred)
#     
#     print('$ deepf(w)')
#     pred_sc_kfw = spectral_kw.fit_predict(AffinityW['deepf_k'])
#     tresult, tpred = evaluate.compute_result(pred=pred_sc_kfw, data=sub_f, true=sub_y, ncluster=params['n_clusters'])
#     Result['SC_knn_fw'] = tresult
#     Pred['SC_knn_fw'].extend(tpred)

    print('# Spectral Clustering (gaussian kernel)')
    print('$ embed_x')
    spectral_ge = SpectralClustering(affinity='rbf', n_clusters=params['n_clusters'], gamma=scale_e, eigen_solver='arpack')
    pred_sc_ge = spectral_ge.fit_predict(sub_x)
    tresult, tpred = evaluate.compute_result(pred=pred_sc_ge, data=sub_x, true=sub_y, ncluster=params['n_clusters'])
    Result['SC_gk_e'] = tresult
    Pred['SC_gk_e'].extend(tpred)
    #spectral_ge.fit(sub_x)
    #AffinityW['SC_gk_e'] = spectral_ge.affinity_matrix_
    
    print('$ deepf')
    spectral_gf = SpectralClustering(affinity='rbf', n_clusters=params['n_clusters'], gamma=scale_f, eigen_solver='arpack')
    pred_sc_gf = spectral_gf.fit_predict(sub_f)
    tresult, tpred = evaluate.compute_result(pred=pred_sc_gf, data=sub_f, true=sub_y, ncluster=params['n_clusters'])
    Result['SC_gk_f'] = tresult
    Pred['SC_gk_f'].extend(tpred)
    #spectral_gf.fit(sub_f)
    #AffinityW['SC_gk_f'] = spectral_gf.affinity_matrix_
    
    # precomputed Affinity Matrix
    print('$ embed_x(w)')
    spectral_gw = SpectralClustering(affinity='precomputed', n_clusters=params['n_clusters'], eigen_solver='arpack')
    pred_sc_gew = spectral_gw.fit_predict(AffinityW['embed_f'])
    tresult, tpred = evaluate.compute_result(pred=pred_sc_gew, data=sub_x, true=sub_y, ncluster=params['n_clusters'])
    Result['SC_gk_ew'] = tresult
    Pred['SC_gk_ew'].extend(tpred)
    
    print('$ deepf(w)')
    pred_sc_gfw = spectral_gw.fit_predict(AffinityW['deepf_f'])
    tresult, tpred = evaluate.compute_result(pred=pred_sc_gfw, data=sub_f, true=sub_y, ncluster=params['n_clusters'])
    Result['SC_gk_fw'] = tresult
    Pred['SC_gk_fw'].extend(tpred)
    
    print('Saving evaluation...')
    saveResult(Result, nround, params, i)  
    #saveAffinity(AffinityW, nround, params, i) 
    Result.clear()
    AffinityW.clear()
    
def run_clustering(features, params, nround, n):
    print('=> load data') 
    dName = params['dataset']
    pmname = params['premodel']
    
    embed_x = load_data_from_csv('../csv/base/'+ dName +'/'+ pmname +'_embed_x.csv', 1)
    label = load_data_from_csv('../csv/base/'+ dName +'/label.csv', 0)
    embed_x, label, features = get_random(embed_x, label, features)
    
    Pred = defaultdict(list)
    
    for i in range(n):
        s = i * 1000
        e = (i+1) * 1000
        sub_x = embed_x[s:e]
        sub_y = label[s:e]
        sub_f = features[s:e]
        run_clustering_in_subset(sub_x, sub_y, sub_f, params, nround, Pred, i)
        
        
    print('=> save prediction...') 
    saveLabel(embed_x, label, Pred, nround, params)

    print('=> visualization...')
    # get the confusion matrix
    evaluate.plot_confusionMatrix1(Pred, label, nround, params)
    evaluate.plot_confusionMatrix2(Pred, label, nround, params)
    Pred.clear()
#     plt.close('all') 
    