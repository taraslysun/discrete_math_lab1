'''Module with Prim's and Kruskal's algorithms'''
import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
from networkx.algorithms import tree
from typing import List
from networkx.algorithms import floyd_warshall_predecessor_and_distance
import time

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


def floyd_algorithm(graph: List, nodes: int) -> List:
    '''
    (List, int) -> List
    Performs Floyd's algorithm to find all pairs shortest path
    '''
    # Потрібні змінні
    inf = float('inf')

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
    
    for u in range (0, nodes):
        if matrix[u][u] < 0:
            print ('Negative cycle detected!')

            return False

    return matrix

def print_result(graph: List, nodes: int) -> 0:
    '''
    (List, int) -> 0
    Prints result of Floyd algorithm
    '''
    result = floyd_algorithm(graph, nodes)

    if not result:
        return

    for i in range (0, len(result)):
        for j in range (0, len(result)):
            if result[i][j] > float('inf'):
                result [i][j] = 'inf'

    for x in range (0, len(result)):
        print (f'Node {x}: distance to vertexes: {result[x]}')

    return 0


if __name__ == '__main__':

    list_of_nodes = [10, 20, 40, 80, 160, 320]
    time_taken = []
    time_taken_1 = []

    for _, elem in enumerate(list_of_nodes):
        G = gnp_random_connected_graph(elem, 1, False)
        amount_of_nod = len(G.nodes)

        start = time.time()
        floyd_algorithm(list(G.edges(data = True)), amount_of_nod)
        end = time.time()
        time_taken.append(end - start)

        start_1 = time.time()
        floyd_warshall_predecessor_and_distance(G)
        end_1 = time.time()
        time_taken_1.append(end_1 - start_1)
    
    plt.plot(time_taken, list_of_nodes, color = 'red')
    plt.plot(time_taken_1, list_of_nodes, color = 'blue')
    plt.xlabel('Time taken')
    plt.ylabel('Num of nodes')
    plt.legend(['My algo', 'Integrated algo'])
    plt.show()
    plt.savefig('example.png')

