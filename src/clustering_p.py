'''
clustering.py: contains all clustering algorithms and result evaluation 
'''
from sklearn.cluster import KMeans, SpectralClustering
import cost 
import evaluate
from save import saveResult, saveLabel, saveAffinity
from util import get_scale
from data import get_random_subset, load_data_from_csv

def run_clustering(features, params, nround, subnum):
    #
    #  RUN GRAPH-BASED ALGORITHMS AND EVALUATION
    #
    print('=> construct affinity matrix W') 
    AffinityW = {}    
    dName = params['dataset']
    pmname = params['premodel']
    # sub scale for test
#     x = load_data_from_csv('../csv/base/'+ dName +'/x.csv', 1)
    embed_x = load_data_from_csv('../csv/base/'+ dName +'/'+ pmname +'_embed_x.csv', 1)
    y = load_data_from_csv('../csv/base/'+ dName +'/label.csv', 0)
    embed_x, y, features = get_random_subset(embed_x, y, features, subnum)
    
        
        
    # calculate scale, sigma value for affinity matrix
#     scale_x = get_scale(x, params['scale_nbr'])
    scale_e = get_scale(embed_x, params['scale_nbr'])
    scale_f = get_scale(features, params['scale_nbr'])
    
 
    # create affinity matrix
    AffinityW['embed'] = cost.full_affinity(embed_x, scale=scale_e)
    AffinityW['deepf'] = cost.full_affinity(features, scale=scale_f)
    print('Affinity Matrix ready!' )
     
    #
    # UNSUPERVISED LEARNING EVALUATION
    #
    print('[=> clustering start...]') 
    Result = {}
    Label = {}
    
    # basedline - K-means
    print('# K-means')
    means_k = KMeans(n_clusters=params['n_clusters'])  
#     print('$ x') 
#     pred_kx = means_k.fit_predict(x)
# #     kmeans_assignments, km = get_cluster_sols(pred_km, ClusterClass=KMeans, n_clusters=params['n_clusters'], init_args={'n_init':10})
#     Result['Kmeans_x'], Label['Kmeans_x'] = evaluate.compute_result(pred=pred_kx, data=x, true=y, ncluster=params['n_clusters'])
    
    print('$ embed_x')
    pred_ke = means_k.fit_predict(embed_x)
    Result['Kmeans_e'], Label['Kmeans_e'] = evaluate.compute_result(pred=pred_ke, data=embed_x, true=y, ncluster=params['n_clusters'])
    
    print('$ deepf')
    pred_kf = means_k.fit_predict(features)
    Result['Kmeans_f'], Label['Kmeans_f']  = evaluate.compute_result(pred=pred_kf, data=features, true=y, ncluster=params['n_clusters'])
    

    # Spectral Clustering
    print('# Spectral Clustering (knn)')
    spectral_k = SpectralClustering(affinity='nearest_neighbors', n_clusters=params['n_clusters'], eigen_solver='arpack')
#     print('$ x')

#     pred_sc_kx = spectral_k.fit_predict(x)
# #     ka_sc, km_sc = get_cluster_sols(pred_sck, ClusterClass=KMeans, n_clusters=params['n_clusters'], init_args={'n_init':10})
#     Result['SC_knn_x'], Label['SC_knn_x'] = evaluate.compute_result(pred=pred_sc_kx, data=x, true=y, ncluster=params['n_clusters'])
#     #spectralk.fit(x)
#     #AffinityW['SC_knn_x'] = spectralk.affinity_matrix_.todense()
    
    print('$ embed_x')
    pred_sc_ke = spectral_k.fit_predict(embed_x)
    Result['SC_knn_e'], Label['SC_knn_e'] = evaluate.compute_result(pred=pred_sc_ke, data=embed_x, true=y, ncluster=params['n_clusters'])

    
    print('$ deepf')
    pred_sc_kf = spectral_k.fit_predict(features)
    Result['SC_knn_f'], Label['SC_knn_f'] = evaluate.compute_result(pred=pred_sc_kf, data=features, true=y, ncluster=params['n_clusters'])
    #spectralk.fit(features)
    #AffinityW['SC_knn_f'] = spectralk.affinity_matrix_.todense()

    print('# Spectral Clustering(gaussian kernel)')
#     print('$ x')
#     spectral_gx = SpectralClustering(affinity='rbf', n_clusters=params['n_clusters'], gamma=scale_x, eigen_solver='arpack')
#     pred_sc_gx = spectral_gx.fit_predict(x)
#     Result['SC_gk_x'], Label['SC_gk_x'] = evaluate.compute_result(pred=pred_sc_gx, data=x, true=y, ncluster=params['n_clusters'])
#     #spectralgk.fit(x)
#     #AffinityW['SC_gk_x'] = spectralgk.affinity_matrix_

    print('$ embed_x')
    spectral_ge = SpectralClustering(affinity='rbf', n_clusters=params['n_clusters'], gamma=scale_e, eigen_solver='arpack')
    pred_sc_ge = spectral_ge.fit_predict(embed_x)
    Result['SC_gk_e'], Label['SC_gk_e'] = evaluate.compute_result(pred=pred_sc_ge, data=embed_x, true=y, ncluster=params['n_clusters'])
    
    print('$ embed_x(w)')
    spectral_gew = SpectralClustering(affinity='precomputed', n_clusters=params['n_clusters'], eigen_solver='arpack')
    pred_sc_gew = spectral_gew.fit_predict(AffinityW['embed'])
    Result['SC_gk_ew'], Label['SC_gk_ew'] = evaluate.compute_result(pred=pred_sc_gew, data=embed_x, true=y, ncluster=params['n_clusters'])
    
    
    print('$ deepf')
    spectral_gf = SpectralClustering(affinity='rbf', n_clusters=params['n_clusters'], gamma=scale_f, eigen_solver='arpack')
    pred_sc_gf = spectral_gf.fit_predict(features)
    Result['SC_gk_f'], Label['SC_gk_f'] = evaluate.compute_result(pred=pred_sc_gf, data=features, true=y, ncluster=params['n_clusters'])
    #spectralgkf.fit(x)
    #AffinityW['SC_gk_f'] = spectralgkf.affinity_matrix_
    
    print('$ deepf(w)')
    spectral_gfw = SpectralClustering(affinity='precomputed', n_clusters=params['n_clusters'], eigen_solver='arpack')
    pred_sc_gfw = spectral_gfw.fit_predict(AffinityW['deepf'])
    Result['SC_gk_fw'], Label['SC_gk_fw'] = evaluate.compute_result(pred=pred_sc_gfw, data=features, true=y, ncluster=params['n_clusters'])
         
    print('=> saving result...')
    #saveAffinity(AffinityW, nround, params)  
    saveResult(Result, nround, params)
    saveLabel(embed_x, y, Label, nround, params)

    print('=> visualization...')
    # get the confusion matrix
    evaluate.plot_confusionMatrix1(Label, y, nround, params)
    evaluate.plot_confusionMatrix2(Label, y, nround, params)
    
#     plt.close('all') 
    
