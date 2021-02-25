from config import TWITTER_DATA_DIR
from sklearn.decomposition import PCA
import os, hdbscan, numpy as np, pickle, time, pandas as pd
import preprocessing


# ## Load Data and Reduce Dimensionality

# In[6]:

with open(os.path.join(TWITTER_DATA_DIR, 'cascade_root_embeddings.pkl'), 'rb') as f:
    corpus_embeddings = pickle.load(f)

X_sbert = np.vstack(corpus_embeddings).T
print('shape:', X_sbert.shape)


# In[7]:

pca_transform = PCA(n_components=70)
transformed_data = pca_transform.fit_transform(X_sbert.T)


# In[9]:

var = pca_transform.explained_variance_ratio_.cumsum()[69]
print('explained variance:', var)


# ## Clustering

algorithm = hdbscan.HDBSCAN
algorithm_kwargs = dict(min_cluster_size=2, allow_single_cluster=False, metric='manhattan',
                       core_dist_n_jobs=4, cluster_selection_epsilon=0)

clusterer = algorithm(**algorithm_kwargs)
start_time = time.time()
labels = clusterer.fit_predict(transformed_data)

with open(os.path.join(TWITTER_DATA_DIR, 'clustering_labels_epsilon_0_pca_70.pkl'), 'wb') as f:
    pickle.dump(labels, f)

print('done, time elapsed:', time.time() - start_time)
