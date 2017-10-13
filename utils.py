import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy import interpolate


def dim_reduction_plot(data, label, block_flag):
    """
    Compute linear PCA and scatter the first two components
    """
  
    PCA_model = TruncatedSVD(n_components=3).fit(data)
    data_PCA = PCA_model.transform(data)
    idxc1 = np.where(label==0)
    idxc2 = np.where(label==1)
    plt.scatter(data_PCA[idxc1,0],data_PCA[idxc1,1],s=80,c='r', marker='^',linewidths = 0, label='light infections')
    plt.scatter(data_PCA[idxc2,0],data_PCA[idxc2,1],s=80,c='y', marker='o',linewidths = 0, label='severe infections')
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.title('PCA of the codes')
    plt.legend(scatterpoints=1,loc='best')
    plt.show(block=block_flag)
  
def ideal_kernel(labels):
    """
    Compute the ideal kernel K
    An entry k_ij = 0 if i and j have different class
    k_ij = 1 if i and j have same class
    """
    K = np.zeros([labels.shape[0], labels.shape[0]])
    
    for i in range(labels.shape[0]):
        k = labels[i] == labels
        k.astype(int)
        K[:,i] = k[:,0]
    return K        
    

def interp_data(X, X_len, restore=False, interp_kind='linear'):
    """
    Interpolate data to match the same maximum length in X_len
    If restore is True, data are interpolated back to their original length
    data are assumed to be time-major
    interp_kind: can be 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
    """
    
    [T, N, V] = X.shape
    X_new = np.zeros_like(X)
    
    # restore original lengths
    if restore:
        for n in range(N):
            t = np.linspace(start=0, stop=X_len[n], num=T)
            t_new = np.linspace(start=0, stop=X_len[n], num=X_len[n])
            for v in range(V):
                x_n_v = X[:,n,v]
                f = interpolate.interp1d(t, x_n_v, kind=interp_kind)
                X_new[:X_len[n],n,v] = f(t_new)
            
    # interpolate all data to length T    
    else:
        for n in range(N):
            t = np.linspace(start=0, stop=X_len[n], num=X_len[n])
            t_new = np.linspace(start=0, stop=X_len[n], num=T)
            for v in range(V):
                x_n_v = X[:X_len[n],n,v]
                f = interpolate.interp1d(t, x_n_v, kind=interp_kind)
                X_new[:,n,v] = f(t_new)
                
    return X_new


def classify_with_knn(train_data, train_labels, test_data, test_labels, k=3):
    """
    Perform classification with knn.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_data, train_labels)
    accuracy = neigh.score(test_data, test_labels)
    pred_labels = neigh.predict(test_data)
    F1 = f1_score(test_labels, pred_labels)
    AUC = roc_auc_score(test_labels, pred_labels)

    return accuracy, F1, AUC

def mse_and_corr(targets, preds, targets_len):
    """
    targets and preds must have shape [time_steps, samples, variables]
    targets_len must have shape [samples,]
    """
    mse_list = []
    corr_list = []
    for i in range(targets.shape[1]):
        len_i = targets_len[i]
        test_data_i = targets[:len_i,i,:]
        pred_i = preds[:len_i,i,:]
        mse_list.append(np.mean((test_data_i-pred_i)**2))
        corr_list.append(np.corrcoef(test_data_i.flatten(), pred_i.flatten())[0,1])
    tot_mse = np.mean(mse_list)
    tot_corr = np.mean(corr_list)
    
    return tot_mse, tot_corr

