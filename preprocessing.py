import os
import pickle
import logging
import pandas as pd
from tqdm import tqdm

from tweet import Tweet
from cascade import Cascade
from config import TWITTER_DATA_DIR
from config import CASCADE_DIR


bandwagon_hashtags = set(['armedforcesflagday', 'goodgovernanceday',
                'modiagain2019', 'bjpkamaljyoti', 'mainbhichowkidar',
                     'bjpvijaysankalpbikerally', 'bogibeelbridge'])

def load_campaign():
    with open(os.path.join(TWITTER_DATA_DIR, 'campaign_tweets_new.pkl'), 'rb') as f:
        campaigns = pickle.load(f)
    return campaigns # dict: hashtag -> tweets

def load_user_data():
    with open(os.path.join(TWITTER_DATA_DIR, 'users.pkl'), 'rb') as f:
        user_df = pickle.load(f)
    return user_df

def process_campaign(campaigns, filter_func=lambda x: True, top_n=None):
    # takes a dict of form {hashtag: campaigns} and returns
    # list of cascades
    temp = []
    for hashtag, tweets in tqdm(campaigns.items()):
        temp += parse_cascades(
                        *process_tweet_ts(tweets.values()),
                        filter_func=filter_func, top_n=top_n)
    return temp

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

def parse_cascades(depth_one_rts, ids_to_tweets,
                   filter_func=lambda x: True, top_n=None):
    """ Turns output of process_tweet_ts into a list of Cascade objects
    Input:
    depth_one_rts (dict: str -> list[dict]): dictionary mapping tweet ids to a list of tweet ids that retweeted
    ids_to_tweets (dict: str -> dict): dictionary mapping tweet ids to full tweet dicts
    filter_func (optional, default: always true): function that filters cascades
    top_n (optional, default: None): int that chooses how many cascades to pick from each campaign
    Output:
    cascades (list[Cascade]): list of Cascade objects
    """
    cascades = []
    for root_id, follower_ids in depth_one_rts.items():
        try:
            root = ids_to_tweets.get(root_id, None)
            retweets = [ids_to_tweets[x] for x in follower_ids if x in ids_to_tweets]
            if root is None:
                continue

            c = Cascade(root, retweets)
            if filter_func(c):
                cascades.append(c)
        except Exception as e:
            print(traceback.format_exc())
            continue
    if top_n:
        return sorted(cascades, key=lambda x: -x.n_retweets)[:top_n]
    else:
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
