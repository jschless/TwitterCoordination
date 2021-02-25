# this script is supposed to calculate how many new people a tweet exposes to the hashtag

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
                print('could not split file for user ', username)
                followers = []
            followers_dict[username] = set(followers)
            if len(followers_dict) > MAX_DICTIONARY_SIZE: # lazy caching
                del followers_dict[next(iter(followers_dict.keys()))]
            return set(followers)
        else:
            return set()

results = {} #ht -> {username -> (# template, # non-template)}

for ht, tweets in tqdm(campaigns.items()):
    ht_results = {}
    sorted_tweets = sorted(tweets.values(), key=lambda x: x['date'])
    prev_tweeters = set()
    exposed = set()
    for tweet in sorted_tweets:
        size_before_tweet = len(exposed)
        user = tweet['username']
        if user not in prev_tweeters:
            exposed.update(get_followers(tweet['username']))
            ht_results[user] = len(exposed) - size_before_tweet
            prev_tweeters.add(user)
    results[ht] = ht_results

import pickle, os
from config import TWITTER_DATA_DIR

with open(os.path.join(TWITTER_DATA_DIR, 'n_newly_exposed_by_user.pkl'), 'wb') as f:
    pickle.dump(results, f)
