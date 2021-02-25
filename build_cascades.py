import preprocessing, cascade, graph_tool.all as gt

def build_cascades(c):
    c.create_network('temporal')
    c.create_network('reverse-temporal')
    c.create_network('flow-graph')
    for i in range(3):
        c.create_network('proportional-followers', suffix=str(i))
        c.create_network('uniform', suffix=str(i))

def create_descendents_prop_map(g):
    if 'n_desc' in g.vp:
        return
    g.vp.n_desc = g.new_vertex_property('int')
    def get_n_descendents(g, v):
        descendents = g.get_out_neighbors(v)
        #print(v, ':', descendents)
        if len(descendents) == 0:
            g.vp.n_desc[v] = 0
        else:
            g.vp.n_desc[v] = len(descendents) + sum([get_n_descendents(g, x) for x in descendents])
        return g.vp.n_desc[v]
    get_n_descendents(g, 0)
    return g.vp.n_desc.a

def create_sv_prop_map(g):
    if 'structural_virality' in g.gp:
        print(f'graph already has structural virality {g.gp.structural_virtality}')
        return
    else:
        gprop = g.new_graph_property("float")
        gprop = structural_virality(g)
        g.gp['structural_virality'] = gprop

def structural_virality(g_directed):
    # sum over all pairwise distances
    g = gt.GraphView(g_directed, directed=False)
    n = g.num_vertices()
    total = sum([gt.shortest_distance(g, i, j) for i in range(n) for j in range(i+1, n)])
    # multiple by two because we only did half the distances and its symmetric
    return 2*total/(n*n-1)

def build_statistics(c):
    def temp(g):
        create_sv_prop_map(g)
        create_descendents_prop_map(g)

    networks = [('temporal', ''), ('reverse-temporal', ''),
               ('proportional-followers', '0'),
                ('uniform', '0'),
                ('proportional-followers', '1'),
                ('uniform', '1'),
                ('proportional-followers', '2'),
                ('uniform', '2')]

    for name, n in networks:
        c.modify_network_and_save(name, apply_func=temp, suffix=n, debug=False)
        temp_g = c.create_network(name, suffix=n)
        if 'structural_virality' not in temp_g.gp:
            print('Not working')

preprocessing.campaign_wide_cascade_apply(build_statistics)

# initially building all cascades
# preprocessing.campaign_wide_cascade_apply(build_cascades)
