import preprocessing
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pickle, os
from config import TWITTER_DATA_DIR
from tqdm import tqdm
import numpy as np

# 17NOV2020: job failed bc of memory shortage on iteration 37

campaigns = preprocessing.load_campaign()
result_dict = {}
embeddings_dict = {}
i = 1
for ht, campaign in tqdm(campaigns.items()):
    i += 1
    if i < 37:
        continue

    result = {}
    unique_tweets = []
    templates = []
    n_templates = 0
    for tweet in campaign.values():
        if tweet['template'] != '':
            n_templates += 1
            if  tweet['template'] not in templates:
                unique_tweets.append(tweet)
                templates.append(tweet['template'])
        if tweet['retweet_id'] == '': # root tweet
            unique_tweets.append(tweet)
        result['unique_tweets'] = unique_tweets
        result['len_unique_tweets'] = len(unique_tweets)
        result['len_unique_templates'] =  len(templates)
        result['len_total_templates'] = n_templates

    model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    corpus_embeddings = model.encode([x['text'] for x in unique_tweets])
    embeddings_dict[ht] = corpus_embeddings

    X_sbert = np.vstack(corpus_embeddings).T
    N_NEIGHBORS = round(len(unique_tweets)/20) # search 5%
    result['n_neighbors'] = N_NEIGHBORS
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='ball_tree').fit(X_sbert.T)
    distances, indices = nbrs.kneighbors(X_sbert.T)
    templates, normals = [], []
    for i, tweet in enumerate(unique_tweets):
        neighbors = indices[i, :]
        tweet_type = 'template' if tweet['template'] != '' else 'normal'
        neighbor_tweets = np.array(unique_tweets)[neighbors]
        n_template = len([x for x in neighbor_tweets if x['template'] != ''])
        if tweet['template'] != '':
            templates.append(n_template)
        else:
            normals.append(n_template)

    result['template_n_template_neighbors'] = templates
    result['normal_n_template_neighbors'] = normals

    global_pct_temp = len(templates)/len(unique_tweets)
    result['global_pct_temp'] = global_pct_temp
    result_dict[ht] = result
    with open(os.path.join(TWITTER_DATA_DIR, 'neighbors_results_2.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)

    with open(os.path.join(TWITTER_DATA_DIR, 'unique_tweet_embeddings_2.pkl'), 'wb') as f:
        pickle.dump(embeddings_dict, f)
