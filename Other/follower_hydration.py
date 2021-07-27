import os
import pickle

import pandas as pd
import tweepy
from tqdm import tqdm

TURKEY_DIR = '/pool001/jschless/turkish_astroturfing'
# df = pd.read_csv(os.path.join(TURKEY_DIR, 'trend_tweets.csv'))

consumer_key = 'vS8BMBqq4GyO2heCjI2esKif6'
consumer_secret = 'cqfxKb0KIll78aTERK19muj3z03Xi6Ao7fRcoYbA7MAz6OcE1f'

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# uids = df.author_id.unique().tolist()
print(os.getcwd())
with open('/home/jschless/whatsapp/TwitterCoordination/authors.pkl', 'rb') as f:
    uids = pickle.load(f)

for uid in tqdm(uids):
    if not str(uid) + '.txt' in os.listdir(os.path.join(TURKEY_DIR, 'follower_info')): 
        with open(os.path.join(TURKEY_DIR, 'follower_info', str(uid)+'.txt'), 'wb') as f:
            try:
                pickle.dump(api.followers_ids(user_id=str(uid)), f)
            except Exception as e:
                print('failed for user ', uid)
                print(e)
