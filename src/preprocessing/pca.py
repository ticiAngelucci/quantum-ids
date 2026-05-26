import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(data, n_components=4):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(data)