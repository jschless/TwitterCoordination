#!/usr/bin/python
import os
import gzip
import numpy as np
import pickle
import graph_tool.all as gt
from config import FOLLOWER_DATA_DIR
from config import CASCADE_DIR
from tweet import Tweet, MissingTweet


class Cascade(object):
    def __init__(self, root, retweets):
        self.root = root
        self.retweets = retweets
        self.n_retweets = len(self.retweets)
        self.missing_retweets = self.root.retweets if self.root.retweets else 0
        self.missing_retweets -= self.n_retweets
        self.temporal_cascade = None
        self.flow_graph = None

    @classmethod
    def from_pickle(cls, pickle_file='example_cascade.pkl'):
        # loads a cascade from a pickle file
        with open(os.path.join(CASCADE_DIR, pickle_file), 'rb') as f:
            root, rt_list = pickle.load(f)
            return cls(root, rt_list)


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
                # print(e)
                followers_dict[user] = set()
        return followers_dict

    def save_cascade_components(self, filename=None):
        if filename is None:
            filename = self.root.id
        with open(os.path.join(CASCADE_DIR, filename + '.pkl'), 'wb') as f:
            pickle.dump((self.root, self.retweets), f)
        return os.path.join(CASCADE_DIR, filename + '.pkl')

    def get_top_tweets(self, thresh=5):
        """ returns a dictionary of tweets and their implied outdegree
        thresh is an int that is the min out degree for a node to be recorded
        """
        if self.temporal_cascade == None:
            self.create_temporal_cascade()

        g = self.temporal_cascade
        locs = np.where(g.get_out_degrees(g.get_vertices()) > thresh)
        return [(g.vp.vertex_to_tweet[v], g.vertex(v).out_degree()) for v in locs[0]]

    def probabilistic_network_construction(self, kind='uniform', log=False):
        """ Creates a retweet network

        type (optional): string that describes type of network to
            'uniform': randomly choose one of the potential parents
            'proportional-followers': proportional to followers
            'temporal': most recent guy is who we pick
            'flow-graph': draw edge between all possibilities

        log (optional): boolean that determines whether to log n_followers in
                        proportional-followers cascade
        """
        followers_dict = self.get_follower_info()
        network, id_dict = self.initialize_cascade_nodes()

        if kind == 'proportional-followers':
            # create a mapping of usernames to followers only once
            follower_counts = {}
            for t in [self.root, *self.retweets]:
                temp = followers_dict.get(t.username, [])
                # for missing users, assume they have 10 followers
                follower_counts[t.username] = len(temp) if len(temp) > 0 else 10

        for i in range(self.n_retweets-1, -1, -1):
            retweeter = self.retweets[i]
            v = id_dict[retweeter.id]
            potential_parents = [self.retweets[j] for j in range(i-1, -1, -1)]
            potential_parents.append(self.root)
            # limit potential parents to following
            potential_parents = [p for p in potential_parents if
                                retweeter.username in followers_dict[p.username]]
            if len(potential_parents) == 0:
                u = id_dict[self.root.id]
                network.add_edge(u, v)
            elif kind == 'uniform':
                parent = np.random.choice(potential_parents)
                u = id_dict[parent.id]
                network.add_edge(u, v)
            elif kind == 'temporal':
                u = id_dict[potential_parents[0].id]
                network.add_edge(u, v)
            elif kind == 'flow-graph':
                for p in potential_parents:
                    u = id_dict[p.id]
                    network.add_edge(u, v)
            elif kind == 'proportional-followers':
                n_followers = [follower_counts[p.username] for p in potential_parents]
                if log:
                    n_followers = np.log(n_followers)
                normed = [x/sum(n_followers) for x in n_followers]
                parent = np.random.choice(potential_parents, p=normed)
                u = id_dict[parent.id]
                network.add_edge(u, v)
            else:
                raise NameError(f'{kind} is not a supported network type')
        return network


    def create_network(self, kind, from_memory=True):
        # wrapper for probabilistic_network_construction
        file_name = os.path.join(CASCADE_DIR, f'{self.root.id}_{kind}.gt')
        if os.path.exists(file_name) and from_memory:
            return gt.load_graph(file_name)
        else:
            g = self.probabilistic_network_construction(kind=kind)
            g.save(file_name)
            return g

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
