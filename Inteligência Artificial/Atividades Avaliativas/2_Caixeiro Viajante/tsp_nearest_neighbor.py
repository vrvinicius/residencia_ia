# %%
import tsplib95 as tsp
import networkx as nx
import matplotlib.pyplot as plt
import math

tsp_problem = "att48.tsp"
file_path = f"C:/Users/vinicius_vieira/OneDrive - Sicredi/Residência IA/Códigos/Inteligência Artificial/Atividades Avaliativas/2_Caixeiro Viajante/data/{tsp_problem}"
problem = tsp.load(file_path)

nodes = list(problem.get_nodes())
edges = list(problem.get_edges())

G = problem.get_graph()


# recalculating edge weights based on Euclidean distance
def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def recalculate_edge_weights(G, node_coords):
    for u, v in G.edges():
        G[u][v]["weight"] = euclidean_distance(node_coords[u], node_coords[v])


node_coords = problem.node_coords  # replacing the nodes for the new calculated ones

recalculate_edge_weights(G, node_coords)


def nearest_neighbor_tsp(G, start_node):
    path = [start_node]
    while len(path) < len(G.nodes):
        last_visited_node = path[-1]
        next_node = min(
            (
                node for node in G.nodes if node not in path
            ),  # all nodes thar are not in the path yet
            key=lambda node: G[last_visited_node][node][
                "weight"
            ],  # return the disacne between the last visited node and the possible next
        )
        path.append(next_node)
    path.append(start_node)  # return to the start node
    return path


def count_tsp_path_nodes(tsp_path):
    return len(tsp_path)


def calculate_total_distance(G, tsp_path):
    total_distance = 0
    for i in range(len(tsp_path) - 1):
        total_distance += G[tsp_path[i]][tsp_path[i + 1]]["weight"]
    return total_distance


# finding the best starting node
best_start_node = None
min_total_distance = float("inf")
best_tsp_path = None

for node in G.nodes:
    tsp_path = nearest_neighbor_tsp(G, node)
    total_distance = calculate_total_distance(G, tsp_path)
    if (
        total_distance < min_total_distance
    ):  # validate if it is the shortest distance possible
        min_total_distance = total_distance
        best_start_node = node
        best_tsp_path = tsp_path

print("Problem: ", tsp_problem)
print("Best start node:", best_start_node)
print("Best TSP Path:", best_tsp_path)
print("Number of nodes in Best TSP Path:", count_tsp_path_nodes(best_tsp_path))
print("Total distance of Best TSP Path:", min_total_distance)
