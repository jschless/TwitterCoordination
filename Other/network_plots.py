from matplotlib.lines import Line2D
import graph_tool.all as gt
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import TWITTER_DATA_DIR

def load_follower_network():
    # loads follower network of all participants in the campaigns
    return gt.load_graph(os.path.join(TWITTER_DATA_DIR, 'full_follower_network_labeled.gt'))
    # return gt.load_graph(os.path.join(TWITTER_DATA_DIR, 'follower_network.gt'))

def label_vertices(g, df):
    zero_exposures = df.query('total_exposure == 0')
    trending = zero_exposures.adj_date > zero_exposures.inferred_trending_start
    after, before = zero_exposures[trending], zero_exposures[~trending]
    templates = df.query('type == "template"')
    regulars = df.query('type == "regular"')
    descriptions = g.new_vertex_property('string')
    n_campaigns = g.new_vertex_property('int')
    for v in tqdm(g.vertices()):
        user = g.vp.usernames[v]
        n_campaigns[v] = len(df[df['username'] == user].hashtag.unique())
        descriptions[v] = 'normal'
        if user in list(before.username):
            descriptions[v] = 'zero-exposure-pre-trending'
        elif user in list(after.username):
            descriptions[v] = 'zero-exposure-post-trending'
        elif user in list(templates.username):
            descriptions[v] = 'template-participant'
        elif user in list(regulars.username):
            descriptions[v] = 'exposed-regular'

    g.vp.descriptions = descriptions
    g.vp.n_campaigns = n_campaigns
    return g

def color_vertices(g, df):
    zero_exposures = df.query('total_exposure == 0')
    trending = zero_exposures.adj_date > zero_exposures.inferred_trending_start
    after, before = zero_exposures[trending], zero_exposures[~trending]
    templates = df.query('type == "template"')
    regulars = df.query('type == "regular"')
    colors = g.new_vertex_property('string')
    for v in g.vertices():
        user = g.vp.usernames[v]
        colors[v] = 'white'
        if user in list(before.username):
            colors[v] = 'red'
        elif user in list(after.username):
            colors[v] = 'blue'
        elif user in list(templates.username):
            colors[v] = 'green'
        elif user in list(regulars.username):
            colors[v] = 'm'
    descriptions = {'Retweets': 'white',
                   'Zero Exposures Pre-Trending': 'red',
                   'Zero Exposures Post-Trending': 'blue',
                   'Template Tweets': 'green',
                   'Regular Tweets': 'm'}

    legend_elements = [
        Line2D([0], [0], marker='o', color='black', label=k,
              markerfacecolor=v, markersize=10)
        for k,v in descriptions.items()]
    fig, ax = plt.subplots()
    ax.legend(handles=legend_elements)

    g.vp.colors = colors
    return g
#g = color_vertices(g, df)

def filter_nodes(g, usernames, include_neighborhood=True, max_neighbors=10,
                largest_component=True):
    # takes large network and filters it down
    # returns a graphview
    v_filter = g.new_vertex_property('bool')
    print('should be this many users', len(usernames))
    for v in g.vertices():
        if g.vp.usernames[v] in usernames:
            v_filter[v] = 1
            if include_neighborhood:
                if v.in_degree() < max_neighbors:
                    for w in v.in_neighbors():
                        v_filter[w] = 1
        else:
            v_filter[v] = 0

    temp = gt.GraphView(g, vfilt=v_filter)

    if largest_component:
        return gt.GraphView(temp, vfilt=gt.label_largest_component(temp))
    else:
        return temp
