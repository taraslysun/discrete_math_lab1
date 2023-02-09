'''Module with Prim's and Kruskal's algorithms'''
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
from networkx.algorithms import tree

def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: int,
                               directed: bool = False,
                               draw: bool = False):
    """
    Generates a random graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted (in case of undirected graphs)
    """


    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    edges = combinations(range(num_of_nodes), 2)
    G.add_nodes_from(range(num_of_nodes))

    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        if random.random() < 0.5:
            random_edge = random_edge[::-1]
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)

    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(-5, 20)
    
    if draw:
        plt.figure(figsize=(10,6))
        if directed:
            # draw with edge weights
            pos = nx.arf_layout(G)
            nx.draw(G,pos, node_color='lightblue', 
                    with_labels=True,
                    node_size=500, 
                    arrowsize=20, 
                    arrows=True)
            labels = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)
            
        else:
            nx.draw(G, node_color='lightblue', 
                with_labels=True, 
                node_size=500)
        
    return G
graph = gnp_random_connected_graph(5,1,False, False)


def prim_algorithm(graph: list[tuple[int, int, dict]]) -> list[tuple[int, int, dict]]:
    '''
    Perform Prim's algorithm to find minimum panning tree
    Graph is given as list of edges with 
    Returns list of tuples with information about minimum planning tree edges
    '''
    pass


def kruskal_algorithm(graph) -> list[tuple[int, int, dict]]:
    '''
    Perform Prim's algorithm to fid minimum panning tree
    Graph is given as list of edges with 
    Returns list of tuples with information about minimum planning tree edges
    '''
    sorted_edges = sorted(graph, key=lambda x: x[2]['weight'])
    nodes = set()
    for edge in sorted_edges:
        nodes.update(edge[:2])
    list_of_nodes = [set([node]) for node in nodes]
    res = []
    while len(list_of_nodes) > 1:
        for edge in sorted_edges:
            node1, node2 = edge[:2]
            for node in list_of_nodes:
                if node1 in node and node2 not in node:
                    list_of_nodes.remove(node)
                    for edg in list_of_nodes:
                        if node2 in edg:
                            list_of_nodes.remove(edg)
                            list_of_nodes.append(node.union(edg))
                            break
                    res.append(edge)
                    break
    return res

# print(graph.edges.data())
print(kruskal_algorithm(list(graph.edges(data=True))))
print(list(tree.minimum_spanning_tree(graph, algorithm="kruskal").edges(data=True)))
