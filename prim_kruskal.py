'''Module with Kruskal's algorithm'''
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


def kruskal_algorithm(graph) -> list[tuple[int, int, dict]]:
    '''
    Perform Prim's algorithm to fid minimum panning tree
    Graph is given as list of edges with 
    Returns list of tuples with information about minimum planning tree edges
    '''
    sorted_edges = sorted(list(graph.edges(data=True)), key=lambda x: x[2]['weight'])
    list_of_nodes = [set([node]) for node in list(graph.nodes())]
    res = []
    while len(list_of_nodes) > 1:
        for edge in sorted_edges:
            for node in list_of_nodes:
                if edge[0] in node and edge[1] not in node:
                    list_of_nodes.remove(node)
                    for node_set in list_of_nodes:
                        if edge[1] in node_set:
                            list_of_nodes.remove(node_set)
                            list_of_nodes.append(node.union(node_set))
                            break
                    res.append(edge)
                    break
    return res

if __name__=="__main__":
    grph = gnp_random_connected_graph(5,1,False,False)
    print(kruskal_algorithm(grph))
    print(list(tree.minimum_spanning_tree(grph, algorithm="kruskal").edges(data=True)))
