import networkx as nx
import matplotlib.pyplot as plt

# Create a new graph
G = nx.Graph()

# read edge index from a file
edge_index = []
with open("./src/experiments/edge_index.txt", "r") as f:
    for line in f:
        edge_index.append(tuple(map(int, line.split())))

# read edge label from a file
edge_label = []
with open("./src/experiments/edge_label.txt", "r") as f:
    for line in f:
        edge_label.append(int(line))

# add edges to the graph
for i, e in enumerate(edge_index):
    # if edge_label[i] == 1:
        G.add_edge(e[0], e[1], length=10, color='green')

# Choose a layout
pos = nx.spring_layout(G)

# Create a larger figure to accommodate the dense graph
plt.figure(figsize=(15, 15))

# Extract edge colors
edge_colors = [G[u][v]['color'] for u, v in G.edges()]

# Draw the graph with the specified edge colors
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=edge_colors, node_size=100, font_size=6, alpha=0.5)

# save the plot
plt.savefig("./src/experiments/graph.png")
