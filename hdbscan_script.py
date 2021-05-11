from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pickle, os
from config import TWITTER_DATA_DIR
from tqdm import tqdm
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
import re, time
from hashtags import hashtags

N_COMPONENTS = 70
EPSILON = 30

result_dict = {}
for ht in tqdm(hashtags):
    X_sbert = np.load(os.path.join(TWITTER_DATA_DIR, 'campaigns', ht+'_embeddings.npy'))

    print('data has shape', X_sbert.shape)
    pca_transform = PCA(n_components=N_COMPONENTS)

    transformed_data = pca_transform.fit_transform(X_sbert.T)
    var = pca_transform.explained_variance_ratio_.cumsum()[N_COMPONENTS-1]
    print('PCA with', N_COMPONENTS, 'has explained variance', var)

    algorithm = hdbscan.HDBSCAN
    algorithm_kwargs = dict(metric='manhattan', cluster_selection_epsilon=EPSILON)

    clusterer = algorithm(**algorithm_kwargs)

    start_time = time.time()
    labels = clusterer.fit_predict(transformed_data)

    print('time spent:', (time.time() - start_time))
    result_dict[ht] = labels

    with open(os.path.join(TWITTER_DATA_DIR, 'campaigns', ht+'_hdbscan_labels_'+str(N_COMPONENTS)+'_'+str(EPSILON)+'.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)
