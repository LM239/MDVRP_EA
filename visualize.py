import sys
import matplotlib.pyplot as plt
import networkx as nx
from math import sqrt
import time

def euclidean_dist(x1, x2, y1, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

if __name__ == "__main__":
    time.sleep(0.01)
    project = sys.argv[1]
    project_def = open("data_files/" + project)

    customers = []
    depots = []
    customer_nodes = []
    depot_nodes = []
    max_vehicles, n_customers, num_depots = [int(val.strip()) for val in project_def.readline().strip().split()]

    for i in range(num_depots):
        project_def.readline()

    for i in range(n_customers):
        data = [int(val.strip()) for val in project_def.readline().strip().split()]
        customers.append((data[1], data[2]))
        customer_nodes.append(((data[1], data[2]), {"color": 'r', "pos": (data[1], data[2])}))

    for i in range(num_depots):
        data = [int(val.strip()) for val in project_def.readline().strip().split()]
        depots.append((data[1], data[2]))
        customer_nodes.append(((data[1], data[2]), {"color": 'b', "pos": (data[1], data[2])}))

    project_def.close()

    routes = open("solution_files/{}.txt".format(project))
    routes.readline()

    G = nx.Graph()
    G.add_nodes_from(customer_nodes)
    G.add_nodes_from(depot_nodes)
    index = 0
    colors = ["b", "g", "r", "c", "m", "y", "k", 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    route_length = 0
    while True:
        data = routes.readline()
        color = colors[index % len(colors)]
        if data == "":
            break
        data = [int(val.strip()) if "." not in val else 0 for val in data.strip().split()]
        depot = data[0] - 1
        route = [val - 1 for val in data][5:-1]
        G.add_edge(depots[depot], customers[route[0]], color=color)
        route_length += euclidean_dist(depots[depot][0], customers[route[0]][0], depots[depot][1], customers[route[0]][1])
        for i in range(0, len(route) - 1):
            G.add_edge(customers[route[i]], customers[route[i + 1]], color=color)
            route_length += euclidean_dist(customers[route[i + 1]][0], customers[route[i]][0], customers[route[i + 1]][1],
                                           customers[route[i]][1])
        G.add_edge(depots[depot], customers[route[-1]], color=color)
        route_length += euclidean_dist(depots[depot][0], customers[route[-1]][0], depots[depot][1],
                                       customers[route[-1]][1])
        index += 1
    routes.close()
    print("Total length of routes: ", route_length)

    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = ["red"] * n_customers
    node_colors.extend(["green"] * num_depots)
    sizes = [75] * n_customers
    sizes.extend([175] * num_depots)
    nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, node_size=sizes)
    fig = plt.gcf()
    fig.savefig("solution_plots/{}.png".format(project))
    plt.show()
