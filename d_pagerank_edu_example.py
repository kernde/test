### Educational example, see damping factor

damping = 0.85

G = nx.DiGraph()
[G.add_node(k) for k in ["A", "B", "C", "D", "E", "G"]]
G.add_edges_from([('G','A'),('B','A'),('A','B'),
                  ('C','A'),('A','C'),('H','D'), ('C','E'),('E','C'),
                  ('E','A'),('A','E'),('D','B'),('B','D')])

fraudsters = {"G":0.2, 'H': 1} # let's say 'h' as done a bigger criminal offence
ppr1 = nx.pagerank(G, personalization=fraudsters, alpha=damping)

pos = nx.spiral_layout(G)
def colfunc(val, minval, maxval, startcolor, stopcolor):
    """ Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the one returned are
        composed of a sequence of N component values (e.g. RGB).
    """
    f = float(val-minval) / (maxval-minval)
    return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))
colfunc(0.2, 0, 1, (1, 0, 0),(0, 1, 0))
pos = nx.spring_layout(G)

nx.draw(G,pos, with_labels = True, node_color=[colfunc(v, 0, max(ppr1.values()),(0, 1, 0),(1, 0, 0)) for v in ppr1.values()],
        nodelist=[k for k in ppr1.keys()], node_size=200)
node_attrs = ppr1 #nx.get_node_attributes(G, 'type')
pos_attrs = {}
for node, coords in pos.items():
    pos_attrs[node] = (coords[0], coords[1] + 0.08)
custom_node_attrs = {}
for node, attr in node_attrs.items():
    custom_node_attrs[node] = str(round(attr,2)*100) + '%'
nx.draw_networkx_labels(G, pos_attrs, labels=custom_node_attrs)

import tikzplotlib
tikzplotlib.save("output_files/pers_pagerank_damp_85.tex")
