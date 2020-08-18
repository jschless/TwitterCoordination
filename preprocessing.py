import pandas as pd
import os
import pickle
from config import TWITTER_DATA_DIR

def load_campaign():
    with open(os.path.join(TWITTER_DATA_DIR, 'campaign_tweets_new.pkl'), 'rb') as f:
        campaigns = pickle.load(f)
    return campaigns # dict: hashtag -> tweets

def load_user_data():
    with open(os.path.join(TWITTER_DATA_DIR, 'users.pkl'), 'rb') as f:
        user_df = pickle.load(f)

def process_tweet_ts(tts):
    """Constructs RT cascades from time series of tweets
    Input:
    tts (list[dict]): list of tweets forming a time series

    Output:
    depth_one_rts (dict: str -> list[dict]): dictionary mapping tweet ids to a list of tweet ids that retweeted
    ids_to_tweets (dict: str -> dict): dictionary mapping tweet ids to full tweet dicts
    """
    tts = sorted(tts, key=lambda x: x['date'])
    depth_one_rts = {}
    ids_to_tweets = {}
    for tweet in tts:
        ids_to_tweets[tweet['id']] = Tweet(tweet)
        rt_id = tweet['retweet_id']
        if rt_id == '': # this is an original tweet
            depth_one_rts[tweet['id']] = []
        else:
            rts = depth_one_rts.get(rt_id, list())
            rts.append(tweet['id'])
            depth_one_rts[rt_id] = rts
    return depth_one_rts, ids_to_tweets

def parse_cascades(depth_one_rts, ids_to_tweets):
    """ Turns output of process_tweet_ts into a list of Cascade objects
    Input:
    depth_one_rts (dict: str -> list[dict]): dictionary mapping tweet ids to a list of tweet ids that retweeted
    ids_to_tweets (dict: str -> dict): dictionary mapping tweet ids to full tweet dicts

    Output:
    cascades (list[Cascade]): list of Cascade objects
    """
    cascades = []
    for root_id, follower_ids in depth_one_rts.items():
        try:
            root = ids_to_tweets[root_id]
            retweets = [ids_to_tweets[x] for x in follower_ids]
            cascades.append(Cascade(root, retweets))
        except:
            continue
            #print('missing info related to root ', root_id, ' or followers ', ','.join(follower_ids))
    return cascades

def parse_cascades_low_mem(depth_one_rts, ids_to_tweets, cascade_func=None):
    """ Processes cascades one at a time to conserve memory
    Input:
    depth_one_rts (dict: str -> list[dict]): dictionary mapping tweet ids to a list of tweet ids that retweeted
    ids_to_tweets (dict: str -> dict): dictionary mapping tweet ids to full tweet dicts
    cascade_func (function): function that operates on cascades and does something you want

    Output:
    None
    """
    cascade = None
    for root_id, follower_ids in tqdm(depth_one_rts.items()):
        try:
            root = ids_to_tweets[root_id]
            retweets = [ids_to_tweets[x] for x in follower_ids]
            cascade = Cascade(root, retweets)
            if cascade_func:
                cascade_func(cascade)
        except Exception as e:
            print(e)
            continue
