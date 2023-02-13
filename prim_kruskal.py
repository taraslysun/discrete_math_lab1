'''Module with Prim's and Kruskal's algorithms'''
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
from networkx.algorithms import tree

from networkx.algorithms import floyd_warshall_predecessor_and_distance

import time
from tqdm import tqdm

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



def prim_algorithm(graph: list[tuple[int, int, dict]]) -> list[tuple[int, int, dict]]:
    '''
    Perform Prim's algorithm to fid minimum panning tree
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
    pass


def floyd_algorithm(graph: list, nodes: int) -> list:
    '''
    Perform Floyd's algorithm to find all pairs shortest path
    '''
    # Потрібні змінні
    inf = 10 ** 3

    # Пуста матриця ваг
    matrix = [[] * nodes] * nodes
    for a in range (0, nodes):
        matrix[a] = [inf] * nodes

    # Заповнена матриця ваг
    for t in graph:
        matrix[t[0]][t[1]] = t[2]['weight']
    
    for u in range (0, nodes):
        matrix[u][u] = 0

    # Аглоритм Флойда
    for k in range (0, nodes):
        for i in range (0, nodes):
            for j in range (0, nodes):
                matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])

    return matrix



if __name__ == '__main__':
    #G = gnp_random_connected_graph(4, 1, True, False)

    #amount_of_nod = len(G.nodes)

    #print(floyd_algorithm(list(G.edges(data = True)), amount_of_nod))

    # print(list(G.edges(data = True)))
    # print(amount_of_nod)

    # pred, dist = floyd_warshall_predecessor_and_distance(G) 
    # for k, v in dist.items():
    #     print(f"Distances with {k} source:", dict(v))

    NUM_OF_ITERATIONS = 1000
    time_taken = 0
    for i in tqdm(range (NUM_OF_ITERATIONS)):
        
        G = gnp_random_connected_graph(100, 0.4, False)
        amount_of_nod = len(G.nodes)
        
        start = time.time()
        floyd_algorithm(list(G.edges(data = True)), amount_of_nod)
        end = time.time()
        
        time_taken += end - start

    print(time_taken / NUM_OF_ITERATIONS)