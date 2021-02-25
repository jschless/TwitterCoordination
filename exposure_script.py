import preprocessing, os, gzip
from config import FOLLOWER_DATA_DIR
import graph_tool.all as gt
from tqdm import tqdm
import network_plots
print('[info] loading campaign info and follower network')

campaigns = preprocessing.load_campaign()
campaigns = {k:v for k,v in campaigns.items() if k not in preprocessing.bandwagon_hashtags}
g = network_plots.load_follower_network()

username_to_index = {g.vp.usernames[i]: i for i in g.vertices()}

def is_following(child, parent):
    child_id = username_to_index.get(child, None)
    parent_id = username_to_index.get(parent, None)
    if child_id is None or parent_id is None:
        return False
    return g.edge(child_id, parent_id) is not None

users = True
if users:
    print('running script to calculate exposure based on exposure to unique users')
else:
    print('running script to calculate exposure based on exposure to tweets')

results = {} #ht -> {username -> (# template, # non-template)}

for ht, tweets in tqdm(campaigns.items()):
    ht_results = {}
    # sort tweets in reverse order
    sorted_tweets = sorted(tweets.values(), key=lambda x: x['date'])[::-1]
    for i in range(len(sorted_tweets)):
        prev_exposures = set()
        child = sorted_tweets[i]
        n_temp, n_norm = 0, 0
        for j in range(i, len(sorted_tweets)):
            # loop through all tweets that happened before
            potential_parent = sorted_tweets[j]

            if is_following(child['username'], potential_parent['username']):
                if users: # if we're tracking unique users
                    if potential_parent['username'] not in prev_exposures:
                        if potential_parent['template'] != '':
                            n_temp += 1
                        else:
                            n_norm += 1
                        prev_exposures.add(potential_parent['username'])
                else: # if not users, we count regardless
                    if potential_parent['template'] != '':
                        n_temp += 1
                    else:
                        n_norm += 1

        ht_results[child['username']] = (n_temp, n_norm, child['date'])
    results[ht] = ht_results

import pickle, os
from config import TWITTER_DATA_DIR

f_name = os.path.join(TWITTER_DATA_DIR, 'exposure_results_n_users.pkl')
with open(f_name, 'wb') as f:
    pickle.dump(results, f)
print('dumped the results into ', f_name)
