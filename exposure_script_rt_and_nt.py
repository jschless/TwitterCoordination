import preprocessing, os, gzip
from config import FOLLOWER_DATA_DIR
import graph_tool.all as gt
from tqdm import tqdm
campaigns = preprocessing.load_campaign()
campaigns = {k:v for k,v in campaigns.items() if k not in preprocessing.bandwagon_hashtags}
followers_dict = {}
MAX_DICTIONARY_SIZE = 10000

def get_followers(username):
    if username in followers_dict:
        return followers_dict[username]
    else:
        file_name = os.path.join(FOLLOWER_DATA_DIR, username + '.gz')
        if os.path.isfile(file_name):
            f = gzip.open(file_name, 'rb')
            try:
                followers = [x.split('\t')[1] for x in f.read().decode().strip().split('\n')]
            except:
                print('error with ', username)
                followers = []
            followers_dict[username] = set(followers)
            if len(followers_dict) > MAX_DICTIONARY_SIZE: # lazy caching
                del followers_dict[next(iter(followers_dict.keys()))]
            return set(followers)
        else:
            return set()

results = {} #ht -> {username -> (# template, # non-template)}

for ht, tweets in tqdm(campaigns.items()):
    ht_results_rt, ht_results_nt = {}, {} # for rts and new tweets
    sorted_tweets = sorted(tweets.values(), key=lambda x: x['date'])[::-1]
    for i in range(len(sorted_tweets)):
        child = sorted_tweets[i]
        n_temp, n_norm = 0, 0
        for j in range(i, len(sorted_tweets)):
        #print(sorted_tweets[j]['id'])
            potential_parent = sorted_tweets[j]
            followers = get_followers(potential_parent['username'])
            if sorted_tweets[j]['username'] in followers:
                if potential_parent['template'] != '':
                    n_temp += 1
                else:
                    n_norm += 1
        if child['retweet_id'] != '':
            ht_results_rt[child['username']] = (n_temp, n_norm, child['date'])
        else:
            ht_results_nt[child['username']] = (n_temp, n_norm, child['date'])
    results[ht] = ht_results_rt, ht_results_nt

import pickle, os
from config import TWITTER_DATA_DIR

with open(os.path.join(TWITTER_DATA_DIR, 'exposure_results_rt_and_nt.pkl'), 'wb') as f:
    pickle.dump(results)
