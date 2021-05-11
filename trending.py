import preprocessing
import pandas as pd
import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date, timedelta
from config import TWITTER_DATA_DIR, FOLLOWER_DATA_DIR, TRENDS_DIR
import pickle
import seaborn as sns
import statsmodels.formula.api as smf
import graphviz as gr
import warnings
import numpy as np
from typing import Tuple
warnings.filterwarnings('ignore')

def get_trend_behavior(ht):
    '''Takes a hashtag and compiles all trend date about it
    Currently just returns the beginning and end of the trending status, but can do more potentially
    '''
    res = []
    for entry in os.scandir(TRENDS_DIR):
        # crude filtering for date csv names
        if entry.path[-7] == '-':
            df = pd.read_csv(entry, header=None,
                            names=['hashtag', 'place',
                                   'level', 'day',
                                   'time', 'volume', 'nan'])
            df['hashtag'] = df.hashtag.apply(lambda x: x.replace('#','').lower())
            try:
                df['datetime'] = pd.to_datetime(df.day + df.time, format='%Y-%m-%d%H:%M')
            except:
                print('error with hashtag', ht)
                return df
            temp = df.query(f'hashtag == "{ht}"')
            if len(temp) > 0:
                res.append(temp)
    if len(res) > 0:
        return pd.concat(res)
    else:
        print(ht, 'did not appear to trend')
        return pd.DataFrame()

def build_df(hashtag, tweet_dict, exposures,
             time_bin='5Min', plot=False, normalize_time=False,
             cutoff_choice='best', include_missing=True,
             periods=1, raw_df_too=False, cumulative=False,
             cache=False, new_trending_data=True, trending_loc_cutoff=50,
             user_exposes_himself=True):
    '''Construct a dataframe with various specifications

    Keyword arguments:
    hashtag: the hashtag you're looking at
    tweet_dict: a dictionary of all tweets in the hashtag
    exposures: a dictionary that converts a hashtag user to their exposure

    normalize_time: whether to reindex so beginning of trending is time 0, otherwise its datetime
    time_bin: how to slice up the time bins using pandas
    cutoff_choice: determines how to infer the exact trending time
    include_missing: whether to include tweets from users we do not have information for.
    periods: when looking for the inflection (cutoff_choice=best), how many periods should you look over?


    '''
    users = set()

    for tweet in sorted(tweet_dict.values(), key=lambda x: x['date']):
        tweet['adj_date'] = tweet['date'] + timedelta(hours=5, minutes=30)
        if tweet['retweet_from'] == '':
            if tweet['template'] != '':
                tweet['type'] = 'template'
            else:
                tweet['type'] = 'regular'
        else:
            if tweet['template'] != '':
                tweet['type'] = 'template_retweet'
            else:
                tweet['type'] = 'regular_retweet'

        f = os.path.join(FOLLOWER_DATA_DIR, tweet['username'] + '.gz')
        tweet['follower_data'] = os.path.isfile(f)

        t_exp, n_exp, _ = exposures[hashtag][tweet['username']]
        already_posted = tweet['username'] in users
        tweet['template_exposure'] = t_exp
        tweet['normal_exposure'] = n_exp
        tweet['total_exposure'] = t_exp + n_exp
        if t_exp + n_exp == 0 and already_posted and user_exposes_himself:
            tweet['total_exposure'] = 1
        users.add(tweet['username'])

    df = pd.DataFrame.from_dict(tweet_dict).transpose()
    df.index = df['adj_date']
    if not include_missing:
        df = df[df.follower_data == True]

    if new_trending_data:
        f_name = os.path.join(TRENDS_DIR, hashtag+'_country_trending.pkl')
        trending_data = pd.read_pickle(f_name)

        # apply query, must be lower than the stated ranking

        trending_data = trending_data[trending_data.within_group_ranking <= trending_loc_cutoff]

        if len(trending_data) == 0:
            print(hashtag, ' did not trend with location less than or equal to',
                  trending_loc_cutoff)
            return None, None
        start = trending_data.datetime.min() + timedelta(hours=5, minutes=30)
        end = trending_data.datetime.max() + timedelta(hours=5, minutes=30)

    else:
        trending_data = pd.read_csv(os.path.join(TRENDS_DIR, hashtag+'.csv'), parse_dates=['datetime'])
        start = trending_data.datetime.min() + timedelta(hours=5, minutes=30)
        end = trending_data.datetime.max() + timedelta(hours=5, minutes=30)
    min_date = start - timedelta(hours=6)
    max_date = end + timedelta(hours=6)

    colors = ["red","orange",'black','grey']
    types = ['template', 'template_retweet', 'regular', 'regular_retweet']

    series_list = []
    for t in types:
        temp = df[df.type == t] # only use regular tweets
        temp = temp.resample(time_bin).count().username.loc[min_date:max_date]
        series_list.append(temp)
    temp = df[df.type == 'regular'] # only use regular tweets

    series_list.append(temp[temp.total_exposure > 1].resample(time_bin).count().username.loc[min_date:max_date])
    series_list.append(temp[temp.total_exposure <= 1].resample(time_bin).count().username.loc[min_date:max_date])

    series_list.append(temp[temp.total_exposure > 0].resample(time_bin).count().username.loc[min_date:max_date])

    temp = temp[temp.total_exposure == 0].resample(time_bin).count().username.loc[min_date:max_date]
    series_list.append(temp)


    new_df = pd.DataFrame(series_list).T

    new_df.columns=[*types, 'greater_1_exposure_regular', 'leq_1_exposure_regular',
                    'nonzero_exposure_regular', 'zero_exposure_regular']
    new_df['total_engagement'] = new_df[types].sum(axis=1)
    new_df['rt_engagement'] = new_df[['regular_retweet', 'template_retweet']].sum(axis=1)

    if cumulative:
        new_df = new_df.cumsum()

    new_df['hashtag'] = hashtag

    if cache:
        # save data strucutres now that we've done the expensive stuff
        return new_df, temp, start


    if len(temp) == 0:
        print('dataframe is empty for some reason', temp)
        print('dates', min_date, max_date)
        return df, df

    exact_trending_loc = find_jump(temp, start, cutoff_choice, periods)

    if normalize_time:
        new_df.index = new_df.index - exact_trending_loc
        new_df.index = new_df.index.map(lambda x: int(x.total_seconds() / 60))
        df.index = df.index - exact_trending_loc
        df.index = df.index.map(lambda x: int(x.total_seconds() / 60))
        exact_trending_loc = 0

    new_df['time'] = new_df.index
    new_df['time_i'] = range(len(new_df))
    new_df['time_i2'] = new_df['time_i']*new_df['time_i']
    new_df['trending_start'] = start
    new_df['inferred_trending_start'] = exact_trending_loc
    new_df = new_df.fillna(0)
    df['time'] = df.index
    df['trending_start'] = start
    df['inferred_trending_start'] = exact_trending_loc
    df['hashtag'] = hashtag

    if plot:
        plot_trending_ts(new_df, exact_trending_loc, hashtag)

    if raw_df_too:
        return new_df, df

    return new_df, exact_trending_loc

def find_jump(x, start, cutoff_choice, periods=1):
    # takes a ts and finds the spike within the start period
    diffed = x.diff(periods=periods)
    #delta = pd.Timedelta(diffed.index.values[1] - diffed.index.values[0])

    try:
        delta = pd.Timedelta(x.index.values[1] - x.index.values[0])
    except Exception as e:
        print(e, 'on this ts\n', x)
    if type(cutoff_choice) is int:
        # manually selecting a period in the range
        return (start - timedelta(hours=1)) + delta*(cutoff_choice-1)

    if cutoff_choice == 'best':
        for i in range(10):
            candidate = diffed.idxmax()
            if candidate >= start-timedelta(hours=1, minutes=5) and candidate < start:
                return candidate - delta
            else: # zero out the candidate
                diffed[candidate] = 0
        # no spike found in trending range
    if cutoff_choice == 'earliest':
        return start - timedelta(hours=1) #- delta

    if cutoff_choice == 'latest':
        return start

    return start #- timedelta(hours=1, minutes=5)

def build_df_cached(new_df, temp, start,
             time_bin='5Min', plot=False, normalize_time=False,
             cutoff_choice='best', include_missing=True,
             periods=1, raw_df_too=False, cumulative=False):

    exact_trending_loc = find_jump(temp, start, cutoff_choice, periods)
    if normalize_time:
        new_df.index = new_df.index - exact_trending_loc
        new_df.index = new_df.index.map(lambda x: int(x.total_seconds() / 60))
        exact_trending_loc = 0

    new_df['time'] = new_df.index
    new_df['time_i'] = range(len(new_df))
    new_df['time_i2'] = new_df['time_i']*new_df['time_i']
    new_df['trending_start'] = start
    new_df['inferred_trending_start'] = exact_trending_loc
    new_df = new_df.fillna(0)

    return new_df, exact_trending_loc


def run_statistics(data, thresh, periods=(12,12),
                   model_str="regular~time_i*threshold+nonzero_exposure_regular",
                   model_fit_args=dict(cov_type='HAC',cov_kwds={'maxlags':1}),
                   plot=False):
    def simple_table_to_df(tab, prefix):
            df = pd.read_html(tab.as_html(), index_col=0, header=0)[0]
            return pd.concat({prefix: df}, names=['Trial'])

    rdd_df = data.assign(threshold=(data.time > thresh).astype(int))
    i_thresh = data.time_i[data.time == thresh].values[0]

    # print('thresh is', thresh, 'i_thresh is', i_thresh, 'value at i_thresh is', rdd_df.iloc[i_thresh])

    # data effects around trending status
    if len(rdd_df.iloc[i_thresh-periods[0]:i_thresh+periods[1]]) == 0:
        rdd_df = rdd_df.iloc[:i_thresh+periods[1]]
        print('Not enough periods in df, running on model from ', rdd_df.iloc[0].time, 'to', rdd_df.iloc[-1].time)
    else:
        rdd_df = rdd_df.iloc[i_thresh-periods[0]:i_thresh+periods[1]]

    try:
        model = smf.wls(model_str, rdd_df).fit(**model_fit_args)
    except Exception as e:
        print(e)
        print('returning df, problem with fitting model, its possible df is empty')
        print(rdd_df)
    if plot:
        ax = rdd_df.plot.scatter(x="time_i", y="regular", color="C0", label='New Non-Template Tweets')
        rdd_df.assign(predictions=model.fittedvalues).plot(x="time_i", y="predictions", ax=ax, color="C1")
        plt.axvline(i_thresh, color='r', ls='--', label='Inferred Trending Time')
        plt.legend()
        plt.title(f"Regression Discontinuity for #{rdd_df.iloc[0].hashtag} (Local Regression)");

    res_df = simple_table_to_df(model.summary().tables[1], 'New Tweets')
    return res_df

def highlight_reg_output(res_df):
    def highlight_significant(s):
        res = 'background-color: yellow' if abs(float(s.z)) >= 1.96 else ''
        return [res for x in s]

    display(res_df.style.apply(highlight_significant, axis=1))


def plot_trending_ts(df, exact_trending_loc, hashtag, cols=['zero_exposure_regular', 'total_engagement'],
                     trending_rankings=None):
    # takes a df and plots the time series with some acoutrements
    fig = plt.figure(figsize=(14,9))

    trending_data = pd.read_csv(os.path.join(TRENDS_DIR, hashtag+'.csv'), parse_dates=['datetime'])

    start = trending_data.datetime.min() + timedelta(hours=5, minutes=30)
    end = trending_data.datetime.max() + timedelta(hours=5, minutes=30)

    if pd.Timedelta(end-start) > timedelta(hours=10):
        print(f'{hashtag} trended for more than 10 hours, defaulting to only plot first 10')
        end = start + timedelta(hours=10)

    min_date = start - timedelta(hours=6)
    max_date = end + timedelta(hours=6)
    for i, col in enumerate(cols):
        ax = plt.subplot(2, 1, i+1)
        temp = df[col]
        label = None if i > 0 else 'Zero Exposure Tweets'

        ax.scatter(temp.index, temp, label='Zero Exposure Tweets')

        ax.axvline(exact_trending_loc, color='r', ls='--', label='Inferred Trending Time')
        _, max_hist_level = ax.get_ylim()
        ax.plot([start - timedelta(hours=1), start], [max_hist_level*1.1]*2, '|--', color='black', alpha=1,
                 label='Resolution Error')
        ax.plot([start, end], [max_hist_level]*2, '-', color='black', alpha=1)
        ax.plot([start, end], [max_hist_level]*2, '|', color='black')

        if i == 0:
            ax.text(start + (end-start)/2, max_hist_level*1.05, f'#{hashtag} trending', fontsize=12, horizontalalignment='center')
        if trending_rankings and col == 'zero_exposure_regular':
            for i in range(len(trending_rankings)):
                ax.text(start+timedelta(hours=i), max_hist_level*1.05, trending_rankings[i])

        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Tweet volume', fontsize=16)
        ax.set_xlim(min_date, max_date)
        ax.set_title(col)
        import matplotlib.dates as mdates
        hours = mdates.HourLocator()
        minutes  = mdates.MinuteLocator(byminute=range(0,61,15))
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_minor_locator(minutes)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        #ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        # ax.grid(True)
        # ax.grid(True, 'minor')


    fig.autofmt_xdate()
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())
#     plt.legend()
    fig.suptitle(f'#{hashtag} Tweets Over Time', size='x-large', weight='bold' )
#     plt.tight_layout()
#     plt.show()
    return fig, ax


def plot_event_study(df, cat: str='zero_exposure_regular', quantiles: Tuple[float, float] = (.025,.975),
                    lower=-120, upper=120, title=None):
    # useful for plotting panel data
    if title is None:
        title = cat
    df = df.loc[(df.index>lower)&(df.index<upper)]
    mean = df.groupby('time')[cat].mean()
    p025 = df.groupby('time')[cat].quantile(quantiles[0])
    p975 = df.groupby('time')[cat].quantile(quantiles[1])
    plt.errorbar(mean.index, mean, xerr=.5, yerr=[mean-p025, p975-mean],
                 fmt='o', capsize=10)
    plt.title('Event Study of ' + title)
    plt.xlabel('Time Since Trending')
    plt.ylabel('Count')
