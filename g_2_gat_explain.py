###" not working, slaiency & sparse generator

from stellargraph.interpretability.saliency_maps import IntegratedGradientsGAT, IntegratedGradients
from stellargraph.interpretability.saliency_maps import GradientSaliencyGAT
import networkx as nx

graph_nodes = list(G.nodes())
all_gen = generator.flow(graph_nodes)
target_nid = 673114 # highest prob
target_idx = graph_nodes.index(target_nid)
target_gen = generator.flow([target_nid])

all_targets = target_encoding.transform(subjects)

y_true = all_targets[target_idx]

y_pred = model.predict(target_gen).squeeze()
class_of_interest = np.argmax(y_pred)
print(
    "target node id: {}, \ntrue label: {}, \npredicted label: {}".format(
        target_nid, y_true, y_pred.round(2)
    )
)

#int_grad_saliency = IntegratedGradientsGAT(model, train_gen, generator.node_list)
int_grad_saliency = IntegratedGradients(model, train_gen)
integrated_node_importance = int_grad_saliency.get_node_importance(
    target_idx, class_of_interest, steps=50
)
#saliency = GradientSaliencyGAT(model, train_gen)

G_ego = nx.ego_graph(G.to_networkx(), target_nid, radius=len(gat.activations))

integrate_link_importance = int_grad_saliency.get_link_importance(
    target_nid, class_of_interest, steps=25
)
print("integrated_link_mask.shape = {}".format(integrate_link_importance.shape))

integrated_node_importance = int_grad_saliency.get_node_importance(
    target_nid, class_of_interest, steps=25
)
print("\nintegrated_node_importance", integrated_node_importance.round(2))
print(
    "integrated self-importance of target node {}: {}".format(
        target_nid, integrated_node_importance[target_idx].round(2)
    )
)
print(
    "\nEgo net of target node {} has {} nodes".format(target_nid, G_ego.number_of_nodes())
)
print(
    "Number of non-zero elements in integrated_node_importance: {}".format(
        np.count_nonzero(integrated_node_importance)
    )
)

sorted_indices = np.argsort(integrate_link_importance.flatten().reshape(-1))
sorted_indices = np.array(sorted_indices)
integrated_link_importance_rank = [(int(k / N), k % N) for k in sorted_indices[::-1]]

topk = 10
print(
    "Top {} most important links by integrated gradients are {}".format(
        topk, integrated_link_importance_rank[:topk]
    )
)

nx.set_node_attributes(G_ego, values={x[0]: {"subject": x[1]} for x in subjects.items()})

node_size_factor = 1e2
link_width_factor = 4

nodes = list(G_ego.nodes())
colors = pd.DataFrame(
    [v[1]["subject"] for v in G_ego.nodes(data=True)], index=nodes, columns=["subject"]
)
colors = np.argmax(target_encoding.transform(colors), axis=1) + 1

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
pos = nx.spring_layout(G_ego)
# Draw ego as large and red
node_sizes = [integrated_node_importance[graph_nodes.index(k)] for k in nodes]
node_shapes = [
    "o" if integrated_node_importance[graph_nodes.index(k)] > 0 else "d" for k in nodes
]
positive_colors, negative_colors = [], []
positive_node_sizes, negative_node_sizes = [], []
positive_nodes, negative_nodes = [], []
# node_size_sclae is used for better visualization of nodes
node_size_scale = node_size_factor / np.max(node_sizes)
for k in range(len(node_shapes)):
    if list(nodes)[k] == target_nid:
        continue
    if node_shapes[k] == "o":
        positive_colors.append(colors[k])
        positive_nodes.append(list(nodes)[k])
        positive_node_sizes.append(node_size_scale * node_sizes[k])

    else:
        negative_colors.append(colors[k])
        negative_nodes.append(list(nodes)[k])
        negative_node_sizes.append(node_size_scale * abs(node_sizes[k]))

cmap = plt.get_cmap("jet", np.max(colors) - np.min(colors) + 1)
nc = nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=positive_nodes,
    node_color=positive_colors,
    cmap=cmap,
    node_size=positive_node_sizes,
    with_labels=False,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    node_shape="o",
)
nc = nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=negative_nodes,
    node_color=negative_colors,
    cmap=cmap,
    node_size=negative_node_sizes,
    with_labels=False,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    node_shape="d",
)
# Draw the target node as a large star colored by its true subject
nx.draw_networkx_nodes(
    G_ego,
    pos,
    nodelist=[target_nid],
    node_size=50 * abs(node_sizes[nodes.index(target_nid)]),
    node_shape="*",
    node_color=[colors[nodes.index(target_nid)]],
    cmap=cmap,
    vmin=np.min(colors) - 0.5,
    vmax=np.max(colors) + 0.5,
    label="Target",
)

edges = G_ego.edges()
# link_width_scale is used for better visualization of links
weights = [
    integrate_link_importance[graph_nodes.index(u), graph_nodes.index(v)]
    for u, v in edges
]
link_width_scale = link_width_factor / np.max(weights)
edge_colors = [
    "red"
    if integrate_link_importance[graph_nodes.index(u), graph_nodes.index(v)] > 0
    else "blue"
    for u, v in edges
]

ec = nx.draw_networkx_edges(
    G_ego, pos, edge_color=edge_colors, width=[link_width_scale * w for w in weights]
)
plt.legend()
plt.colorbar(nc, ticks=np.arange(np.min(colors), np.max(colors) + 1))
plt.axis("off")
plt.show()