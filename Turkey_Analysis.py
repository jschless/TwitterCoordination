
# coding: utf-8


# In[1]:


import os 
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

DATA_DIR = '/pool001/jschless/kiran-data/kiran-data/'
TURKEY_DIR = '/pool001/jschless/turkish_astroturfing'


# In[2]:


# Loading df created in the Turkey_Data_Wrangling notebook
df = pd.read_pickle(os.path.join(DATA_DIR, 'mega_df.pkl'))

df = df.drop_duplicates(subset=['id'])


# In[3]:


# get all users that posted a lexicon tweet
astrobots = set(df.query('lexicon == True').author_id.unique())


# In[4]:



def build_df(df, trend,
             include_missing=False, 
             drop_astrobots=True,
             time_bin='5Min', 
             minutes_to_round=5):
    # takes the dataframe and a hashtag and converts it to a event study dataframe for that hashtag

    df = df[df.trend == trend].set_index('created_at')
        
    if not include_missing: 
        # Remove tweets from users lacking follower info
        # df = df.query('author_id != -1 and author != "missing"')
        df = df[df.follower_data == True]

    if drop_astrobots:
        # Remove tweets from users that posted lexicon tweets
        df = df[~df.author_id.isin(astrobots)]

    start, end = df.iloc[0].tr_start, df.iloc[0].tr_end
    start = start - timedelta(minutes=start.minute % minutes_to_round,
                              seconds=start.second,
                              microseconds=start.microsecond)

    df['tweet_type'] = df.text.apply(lambda x: 'retweet' if x.startswith('RT') else 'tweet')
    
    # Binning various statistics of interest
    series_list = []
    
    
    types = ['tweet', 'retweet']    
    # bin tweets and retweets separately
    for t in types:
        series_list.append(
            df[df.tweet_type == t]
              .resample(time_bin)
              .count()
              .author_id
        )
        
    # add in the tweets from unexposed users
    series_list.append(
        df.query('tweet_type == "tweet" and exposed == False')
          .resample(time_bin)
          .count()
          .author_id
    )
        
    new_df = pd.DataFrame(series_list).T
    new_df.columns = [*types, 'zero_exposure_tweets']
    
    # normalize time to minutes before / after trend
    new_df['absolute_time'] = new_df.index
    new_df.index = new_df.index - start
    new_df.index = new_df.index.map(lambda x: int(x.total_seconds() / 60))

    # add additional info to dataset
    new_df['trend'] = trend
    new_df['time'] = new_df.index
    new_df['time_i'] = range(len(new_df))
    new_df['trending_start'] = start

    return new_df


# ### Selecting Specific Hashtags
# There are a lot of hashtags, so it might be worthwhile to remove the hashtags at either extreme. 

# In[7]:


sizes = df.groupby('trend').size()
lower, upper = .25, .75 # thresholds for which quantiles to include
l, u = sizes.quantile([lower, upper])

# only select trends between the lower and upper thresholds
trends = sizes.where(sizes <= u).where(sizes >= l).dropna().index 


# In[8]:


# Include all hashtags
trends = sizes.index


# ## Building Panel
# For each selected trend, build the panel

# In[ ]:


dfs = []
for trend in tqdm(trends):
    try:
        temp_df = build_df(df, trend, time_bin='5Min', minutes_to_round=5)

        if not temp_df is None:
            # assign dummy variable for whether a hashtag is trending or not
            temp_df = temp_df.assign(threshold=(temp_df.time > 0).astype(int))
            dfs.append(temp_df)

    except Exception as e:
        print('Error with', trend, e)

panel_df = pd.concat(dfs)

panel_df.to_pickle(os.path.join(DATA_DIR, 'full_panel_df.pkl'))
