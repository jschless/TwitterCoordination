#!/usr/bin/python
import os
import gzip
import numpy as np
import graph_tool.all as gt
from config import FOLLOWER_DATA_DIR
from config import CASCADE_DIR
from tweet import Tweet, MissingTweet

class Cascade(object):
    def __init__(self, root, retweets):
        self.root = root
        self.retweets = retweets
        self.n_retweets = len(self.retweets)
        self.missing_retweets = self.root.retweets - self.n_retweets
        self.temporal_cascade = None

    def get_follower_info(self):
        # creates dictionary that maps usernames to a set of followers
        followers_dict = {} # username -> set of followers
        for tweet in [self.root, *self.retweets]:
            user = tweet.username
            try:
                file_name = os.path.join(FOLLOWER_DATA_DIR, user + '.gz')
                if os.path.isfile(file_name):
                    f = gzip.open(file_name, 'rb')
                    followers = [x.split('\t')[1] for x in f.read().decode().strip().split('\n')]
                    followers_dict[user] = set(followers)
                else:
                    followers_dict[user] = set()
            except Exception as e:
                # account may be deleted
                print(e)
                followers_dict[user] = set()
        return followers_dict


    def get_top_tweets(self, thresh=5):
    # returns a dictionary of tweets and their implied outdegree
        if self.temporal_cascade == None:
            self.create_temporal_cascade()

        g = self.temporal_cascade
        locs = np.where(g.get_out_degrees(g.get_vertices()) > thresh)
        return [(g.vp.vertex_to_tweet[v], g.vertex(v).out_degree()) for v in locs[0]]



    def network_construction(self, temporal=True):
        followers_dict = self.get_follower_info()
        network, id_dict = self.initialize_cascade_nodes()
        for i in range(self.n_retweets-1, -1, -1):
            retweeter = self.retweets[i]
            v = id_dict[retweeter.id]
            for j in range(i-1, -1, -1):
                prior_retweeter = self.retweets[j]
                if retweeter.username in followers_dict[prior_retweeter.username]:
                    u = id_dict[prior_retweeter.id]
                    network.add_edge(u, v)
                    if temporal:
                        break
                elif j == 0: # we've reached the end, so root is influencer
                    u = id_dict[self.root.id]
                    network.add_edge(u, v)

        return network

    def create_temporal_cascade(self):
        file_name = os.path.join(CASCADE_DIR, f'{self.root.id}_temporal.gt')
        if os.path.exists(file_name):
            self.temporal_cascade = gt.load_graph(file_name)
        else:
            self.temporal_cascade = self.network_construction()
            self.temporal_cascade.save(file_name)
        return self.temporal_cascade

    def create_flow_graph(self):
        self.flow_graph = self.network_construction(temporal=False)
        return self.flow_graph

    def create_follower_network(self):
        self.follower_network, id_dict = self.initialize_cascade_nodes(usernames=True)
        followers_dict = self.get_follower_info()
        for target, sources in followers_dict.items():
            v = id_dict[target]
            for source in list(sources):
                if source in id_dict: # if source was in RT network
                    self.follower_network.add_edge(id_dict[source], v)

        return self.follower_network

    def initialize_cascade_nodes(self, usernames=False):
        # returns graph-tool graph with only nodes and corresponding tweet info
        # nodes are usernames if usernames=True
        g = gt.Graph()
        vertex_to_tweet = g.new_vertex_property('object')
        tweet_id_to_vertex = {}
        possible_missing_tweets = []
        nodes = [self.root, *self.retweets]
        # if usernames:
            # remove duplicates from a single source
        for node in [self.root, *self.retweets]:
            v = g.add_vertex()
            vertex_to_tweet[v] = node
            key = node.username if usernames else node.id
            tweet_id_to_vertex[key] = v
            if node.retweet_from != '':
                possible_missing_tweets.append(
                    (node.retweet_id, node.retweet_from))

        for id, user in possible_missing_tweets:
            # consider filling these deleted tweets in time series
            if id not in tweet_id_to_vertex:
                tweet = MissingTweet(id, user)
                v = g.add_vertex()
                vertex_to_tweet[v] = node
                key = node.username if usernames else node.id
                tweet_id_to_vertex[key] = v


        # internalize property map
        g.vp.vertex_to_tweet = vertex_to_tweet
        return g, tweet_id_to_vertex

    def __repr__(self):
        return f'cascade of size {self.n_retweets}'
