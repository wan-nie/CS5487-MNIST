import numpy as np
import os
import pandas as pd
from model import KNN, SVM, LR
import random
from types import SimpleNamespace


# initialize random seed
random.seed(42); np.random.seed(42)


if __name__ == '__main__':
    data_dir = './data/processed_data'
    X = np.load(os.path.join(data_dir, 'vectors.npy'))
    X = X.reshape(X.shape[0], -1) / 255.
    y = np.load(os.path.join(data_dir, 'labels.npy'))
    c_X = np.load(os.path.join(data_dir, 'cvectors.npy'))
    c_X = c_X.reshape(c_X.shape[0], -1) / 255.
    c_y = np.load(os.path.join(data_dir, 'clabels.npy'))

    data1 = {
        'train_x': X[:2000],
        'test_x': X[2000:],
        'train_y': y[:2000],
        'test_y': y[2000:],
        'c_X': c_X,
        'c_y': c_y,
    }
    data1 = SimpleNamespace(**data1)

    data2 = {
        'train_x': X[2000:],
        'test_x': X[:2000],
        'train_y': y[2000:],
        'test_y': y[:2000],
        'c_X': c_X,
        'c_y': c_y
    }
    data2 = SimpleNamespace(**data2)

    PCA_KERNELS = [None, 'rbf', 'sigmoid']
    PCA_DIMS = [None, 8, 16, 32, 64, 128]

    """
    ############################ KNN ############################
    """
    # find the best hyperparams by 5-fold CV on training set
    k = 1
    metric_ls = ['euclidean', 'manhattan', 'cosine']
    result_ls = []
    for metric in metric_ls:
        for pca_kernel in PCA_KERNELS:
            for pca_dim in PCA_DIMS:
                knn = KNN(n_neighbors=k, metric=metric, pca_kernel=pca_kernel, pca_dim=pca_dim)
                score1, score2 = knn.five_fold(data1, data2)
                result_ls.append([k, metric, pca_kernel, pca_dim, score1, score2])

    result_df = pd.DataFrame(result_ls, columns=['n_neighbors', 'metric', 'pca_kernel', 'pca_dim', 'score1', 'score2'])
    result_df.to_csv('./result/knn_5-fold.csv', index=False)

    # evaluation on test set and challenge test set
    k = 1
    metric = 'cosine'
    pca_kernel = None
    pca_dim = 32
    knn = KNN(n_neighbors=k, metric=metric, pca_dim=pca_dim, pca_kernel=pca_kernel)
    score, c_score = knn.eval(data1)
    print('KNN data1: {:.3f} {:.3f}'.format(score, c_score))

    k = 1
    metric = 'cosine'
    pca_kernel = None
    pca_dim = 64
    knn = KNN(n_neighbors=k, metric=metric, pca_dim=pca_dim, pca_kernel=pca_kernel)
    score, c_score = knn.eval(data2)
    print('KNN data2: {:.3f} {:.3f}'.format(score, c_score))

    """
    ############################ SVM ############################
    """
    # find the best hyperparams by 5-fold CV on training set
    result_ls = []
    svm_kernel_ls = ['poly', 'rbf']
    for svm_kernel in svm_kernel_ls:
        for pca_kernel in PCA_KERNELS:
            for pca_dim in PCA_DIMS:
                svm = SVM(svm_kernel=svm_kernel, pca_kernel=pca_kernel, pca_dim=pca_dim)
                score1, score2 = svm.five_fold(data1, data2)
                result_ls.append([svm_kernel, pca_kernel, pca_dim, score1, score2])
    result_df = pd.DataFrame(result_ls, columns=['svm_kernel', 'pca_kernel', 'pca_dim', 'score1', 'score2'])
    result_df.to_csv('./result/svm_5-fold.csv', index=False)

    # evaluation on test set and challenge test set
    svm_kernel = 'rbf'
    pca_kernel = None
    pca_dim = 32
    svm = SVM(svm_kernel=svm_kernel, pca_kernel=pca_kernel, pca_dim=pca_dim)
    score, c_score = svm.eval(data1)
    print('SVM data1: {:.3f} {:.3f}'.format(score, c_score))

    svm_kernel = 'rbf'
    pca_kernel = None
    pca_dim = 64
    svm = SVM(svm_kernel=svm_kernel, pca_kernel=pca_kernel, pca_dim=pca_dim)
    score, c_score = svm.eval(data2)
    print('SVM data2: {:.3f} {:.3f}'.format(score, c_score))

    """
    ############################ LogisticRegression ############################
    """
    # find the best hyperparams by 5-fold CV on training set
    result_ls = []
    for pca_kernel in PCA_KERNELS:
        for pca_dim in PCA_DIMS:
            lr = LR(pca_kernel=pca_kernel, pca_dim=pca_dim)
            score1, score2 = lr.five_fold(data1, data2)
            result_ls.append([pca_kernel, pca_dim, score1, score2])
    result_df = pd.DataFrame(result_ls, columns=['pca_kernel', 'pca_dim', 'score1', 'score2'])
    result_df.to_csv('./result/lr_5-fold.csv', index=False)

    # evaluation on test set and challenge test set
    pca_kernel = None
    pca_dim = None
    lr = LR(pca_kernel=pca_kernel, pca_dim=pca_dim)
    score, c_score = lr.eval(data1)
    print('LR data1: {:.3f} {:.3f}'.format(score, c_score))

    pca_kernel = None
    pca_dim = None
    lr = LR(pca_kernel=pca_kernel, pca_dim=pca_dim)
    score, c_score = lr.eval(data2)
    print('LR data2: {:.3f} {:.3f}'.format(score, c_score))






