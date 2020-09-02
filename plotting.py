import matplotlib.pyplot as plt

participant_to_color = {'6_non-participant': 6, '2_official': 2, '3_fan': 3,
                        '1_itcell': 1, '5_deleted': 5, '4_other': 4}

def plot_temporal_cascade(g, text_label_thresh=5, output_name=None):
    """ Plots a temporal cascade
    Input:
    g (gt.Graph): graph to plot

    Optional inputs:
    output_name (string): name for output file
    text_label_thresh (int): if out degree > this, username will be labeled
    """
    vtext = g.new_vertex_property('string')
    for v in g.vertices():
        vtext[v] = ''
        if v.out_degree() > text_label_thresh:
            vtext[v] = g.vp.vertex_to_tweet[v].username

    vcolor = color_nodes(g)
    out_deg = g.new_vertex_property('int')
    out_deg.a = np.log(g.get_out_degrees(g.get_vertices())+2)
    output_file = f'plots/{output_name}.pdf' if output_name else 'plots/temp.pdf'
    gt.graphviz_draw(g, layout='dot', output=output_file,
                    vcolor=vcolor, vcmap=matplotlib.cm.jet,
                    vsize=out_deg,
                    vprops={"label": vtext,'fontsize': 100,
                    'fontcolor': 'white'}, size=(60,60))
    return WImage(filename=output_file)

def color_nodes(g):
    # returns a VertexPropertyMap with colorings by a user attribute
    colors = g.new_vertex_property('int')
    for v in g.vertices():
        tweet = g.vp.vertex_to_tweet[v]
        if tweet.username in user_df.index:
            colors[v] = participant_to_color[user_df.loc[tweet.username].type]
    return colors

def show_color_bar(dic, cmap=plt.cm.jet):
    labels = [x[2:] for x in sorted(dic.keys())]
    fig, ax = plt.subplots(1, 1, figsize=(.5, 6))  # setup the plot

    # define the bins and normalize
    bounds = np.linspace(0, len(labels), 7)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
        spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    new_tick_locs = [x+.5 for x in bounds]
    cb.set_ticks(new_tick_locs[:-1])
    cb.set_ticklabels(labels)
