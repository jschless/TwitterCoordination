from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pickle, os
from config import TWITTER_DATA_DIR
from tqdm import tqdm
import numpy as np
from hdbscan import HDBSCAN
import re
from hashtags import hashtags

def process_text(text):
    # remove link
    text = re.sub(r'http\S+', '', text)

    # remove hashtags
    text = re.sub(r'#(\w+)', '', text)

    # remove tags
    text = re.sub(r'@(\w+)', '', text)

    if text.isspace() or text == "": # string is empty
        return None
    return text


with open(os.path.join(TWITTER_DATA_DIR, 'campaign_tweets_new.pkl'), 'rb') as f:
    campaigns = pickle.load(f)
result_dict = {}
raw_tweets = {}
for ht, tweets in tqdm(campaigns.items()):
    result = {}
    unique_tweets = []
    templates = []
    n_templates = 0
    for tweet in tweets.values():
        if tweet['template'] != '':
            n_templates += 1
            if  tweet['template'] not in templates:
                unique_tweets.append(tweet)
                templates.append(tweet['template'])
        elif tweet['retweet_id'] == '': # root tweet
            unique_tweets.append(tweet)

    del templates

    #model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

    for x in unique_tweets:
        x['processed_text'] = process_text(x['text'])


    filtered_unique_tweets = [x for x in unique_tweets if x['processed_text'] is not None]
    raw_tweets[ht] = filtered_unique_tweets
    with open(os.path.join(TWITTER_DATA_DIR, 'processed_raw_tweets.pkl'), 'wb') as f:
         pickle.dump(raw_tweets, f)
    continue

    del unique_tweets

    strings = [x['processed_text'] for x in filtered_unique_tweets]

    result['len_parsed_tweets'] = len(strings)

    if os.path.exists(os.path.join(TWITTER_DATA_DIR, 'campaigns', ht+'_embeddings.npy')):
        X_sbert = np.load(os.path.join(TWITTER_DATA_DIR, 'campaigns', ht+'_embeddings.npy'))
    else:
        model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
        corpus_embeddings = model.encode(strings, show_progress_bar=True)
        X_sbert = np.vstack(corpus_embeddings).T
        np.save(os.path.join(TWITTER_DATA_DIR, 'campaigns', ht+'_embeddings.npy'), X_sbert)


    N_NEIGHBORS = round(len(filtered_unique_tweets)/100) # search 1%
    result['n_neighbors'] = N_NEIGHBORS
    nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='ball_tree').fit(X_sbert.T)
    distances, indices = nbrs.kneighbors(X_sbert.T)
    templates, normals = [], []
    template_tweets, normal_tweets = [], []
    for i, tweet in enumerate(filtered_unique_tweets):
        neighbors = indices[i, :]
        tweet_type = 'template' if tweet['template'] != '' else 'normal'
        neighbor_tweets = np.array(filtered_unique_tweets)[neighbors]
        n_template = len([x for x in neighbor_tweets if x['template'] != ''])
        if tweet['template'] != '':
            templates.append(n_template)
            template_tweets.append((tweet, neighbor_tweets))
        else:
            normals.append(n_template)
            normal_tweets.append((tweet, neighbor_tweets))

    result['template_n_template_neighbors'] = templates
    result['normal_n_template_neighbors'] = normals
    result['template_tweets'] = template_tweets
    result['normal_tweets'] = normal_tweets

    global_pct_temp = len(templates)/len(strings)
    result['global_pct_temp'] = global_pct_temp
    result_dict[ht] = result

    with open(os.path.join(TWITTER_DATA_DIR, 'neighbors_results_1pct.pkl'), 'wb') as f:
         pickle.dump(result_dict, f)


    # with open(os.path.join(TWITTER_DATA_DIR, 'campaigns', ht+'_neighbors_results.pkl'), 'wb') as f:
    #     pickle.dump(result, f)
