'''
save.py: contains all save functions for all type of data (dataset, model, evaluation)
'''
import pandas as pd
from keras.utils import plot_model


def save_data_to_csv(data, path):
    save = pd.DataFrame(data)  
    save.to_csv(path, index=False, sep=',', header=None) 
    
def saveModel(model, path):
    print('=> saving model...')
    print('Model summary:')
    model.summary()
    plot_model(model, to_file=path + '.png')
    model.save(path +'.h5')
    # save Model to json
    model_json = model.to_json()
    with open(path +'.json', 'w') as file:
        file.write(model_json)
    # save Model weights
    model.save_weights(path +'Weights.h5') 

def saveAffinity(W, nround, dname, n): 
    save_data_to_csv(W['embed'], '../csv/affinity/'+ dname +'/'+ str(nround) +'_Affinity_embed_'+ str(n) +'.csv')
    save_data_to_csv(W['deepf'], '../csv/affinity/'+ dname +'/'+ str(nround) +'_Affinity_deepf_'+ str(n) +'.csv')

    save_data_to_csv(W['SC_knn_e'], '../csv/affinity/'+ dname +'/'+ str(nround) +'_Affinity_SC_knn_e_'+ str(n) +'.csv')
    save_data_to_csv(W['SC_knn_f'], '../csv/affinity/'+ dname +'/'+ str(nround) +'_Affinity_SC_knn_f_'+ str(n) +'.csv')
    save_data_to_csv(W['SC_gk_e'], '../csv/affinity/'+ dname +'/'+ str(nround) +'_Affinity_SC_gk_e_'+ str(n) +'.csv')
    save_data_to_csv(W['SC_gk_f'], '../csv/affinity/'+ dname +'/'+ str(nround) +'_Affinity_SC_gk_f_'+ str(n) +'.csv')
    print('Affinity matrix W saved!') 
    
def saveResult(result, nround, params, n):
    df = pd.DataFrame({'Kmeans_e':result['Kmeans_e'], 'Kmeans_f':result['Kmeans_f'],
                       'SC_knn_e':result['SC_knn_e'], 'SC_knn_f':result['SC_knn_f'], #'SC_knn_ew':result['SC_knn_ew'], 'SC_knn_fw':result['SC_knn_fw'],
                       'SC_gk_e':result['SC_gk_e'], 'SC_gk_f':result['SC_gk_f'], 'SC_gk_ew':result['SC_gk_ew'], 'SC_gk_fw':result['SC_gk_fw']})
    df1 = df.stack()
    df2 = df1.unstack(0)
    df2.columns=['sc','ch','dbi','ARI','FMI','NMI','ACC']
    df2.to_csv('../csv/pred/'+ params['dataset'] +'/'+ params['premodel'] +'/'+ str(nround) +'_result_'+ str(n) +'.csv', index=False, sep=',')
    print('Evaluation saved!')
    
def saveLabel(embed_x, y, label, nround, params):
    df = pd.DataFrame({'Kmeans_e':label['Kmeans_e'], 'Kmeans_f':label['Kmeans_f'],
                       'SC_knn_e':label['SC_knn_e'], 'SC_knn_f':label['SC_knn_f'], #'SC_knn_ew':label['SC_knn_ew'], 'SC_knn_fw':label['SC_knn_fw'],
                       'SC_gk_e':label['SC_gk_e'], 'SC_gk_f':label['SC_gk_f'], 'SC_gk_ew':label['SC_gk_ew'], 'SC_gk_fw':label['SC_gk_fw']})
    df.to_csv('../csv/pred/'+ params['dataset'] +'/'+ params['premodel'] +'/'+ str(nround) +'_label.csv', index=False, sep=',')  
    save_data_to_csv(embed_x, '../csv/pred/'+ params['dataset'] +'/'+ params['premodel'] +'/'+ str(nround) +'_embed_x.csv')
    save_data_to_csv(y, '../csv/pred/'+ params['dataset'] +'/'+ params['premodel'] +'/'+ str(nround) +'_y.csv')
    print('Label saved!')
