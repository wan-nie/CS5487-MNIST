import numpy as np
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from abc import ABC, abstractmethod


class ModelBase(ABC):
    def __init__(self, pca_kernel, pca_dim):
        self.pca_kernel = pca_kernel
        self.pca_dim = pca_dim
        self.model = None

    @abstractmethod
    def get_model_cv(self):
        pass

    def process_data(self, data):

        def pca(train_x, test_x, c_X):
            if self.pca_kernel is None:
                pca = PCA(n_components=self.pca_dim)
            else:
                pca = KernelPCA(n_components=self.pca_dim, kernel=self.pca_kernel)
                # standardize
                scaler = StandardScaler()
                scaler.fit(train_x)
                train_x = scaler.transform(train_x)
                test_x = scaler.transform(test_x)
                if c_X is not None:
                    c_X = scaler.transform(c_X)

            train_x = pca.fit_transform(train_x)
            test_x = pca.transform(test_x)
            if c_X is not None:
                c_X = pca.transform(c_X)

            return train_x, test_x, c_X

        data = copy.deepcopy(data)
        # pca
        if self.pca_dim is not None:
            data.train_x, data.test_x, data.c_X = pca(data.train_x, data.test_x, data.c_X)

        return data

    def five_fold(self, data1, data2):
        data1 = self.process_data(data1)
        data2 = self.process_data(data2)

        def once(X, y):
            ls = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_idx in kf.split(X):
                train_x, val_x = X[train_idx], X[val_idx]
                train_y, val_y = y[train_idx], y[val_idx]
                model_cv = self.get_model_cv()
                model_cv.fit(train_x, train_y)
                score = model_cv.score(val_x, val_y)
                ls.append(score)
            return np.mean(ls)

        score1 = once(data1.train_x, data1.train_y)
        score2 = once(data2.train_x, data2.train_y)

        return score1, score2

    def eval(self, data):
        data = self.process_data(data)

        # trial1 & trial2
        self.model.fit(data.train_x, data.train_y)
        score = self.model.score(data.test_x, data.test_y)
        c_score = self.model.score(data.c_X, data.c_y)  # challenge data

        return score, c_score


class KNN(ModelBase):
    def __init__(
            self,
            n_neighbors,
            metric="minkowski",
            pca_kernel=None,
            pca_dim=None,
    ):
        super().__init__(pca_kernel=pca_kernel, pca_dim=pca_dim)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

    def get_model_cv(self):
        return KNeighborsClassifier(n_neighbors=self.n_neighbors, metric=self.metric)


class SVM(ModelBase):
    def __init__(
            self,
            svm_kernel,
            pca_kernel=None,
            pca_dim=None
    ):
        super().__init__(pca_kernel=pca_kernel, pca_dim=pca_dim)
        self.svm_kernel = svm_kernel
        self.model = SVC(kernel=svm_kernel)

    def get_model_cv(self):
        return SVC(kernel=self.svm_kernel)


class LR(ModelBase):
    def __init__(
            self,
            pca_kernel=None,
            pca_dim=None
    ):
        super().__init__(pca_kernel=pca_kernel, pca_dim=pca_dim)
        self.model = LogisticRegression(
            solver='lbfgs',
            multi_class='multinomial',
            max_iter=10000
        )

    def get_model_cv(self):
        return LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)







