import os
from shutil import copyfile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import choice


class AntResult:
    def __init__(self, my_k, my_je, ant_id):
        self.my_k = my_k
        self.my_je = my_je
        self.ant_id = ant_id


# Prawdopodobieństwo wybrania węzła j będąc na węźle i
def get_p_k_ij(j, i, my_tau, my_alfa, my_metric, my_beta):
    return pow(my_tau[j][i], my_alfa) * pow(get_eta_ij(j, i, my_metric), my_beta) / new_sum_for_get_p_k_ij(my_metric, i,
                                                                                                           my_alfa,
                                                                                                           my_beta,
                                                                                                           my_tau)


def new_sum_for_set_tau_ij(N, my_tau_all, i, j):
    sum = 0
    for k in range(0, N):
        sum += my_tau_all[k][j][i]
    return sum


def new_sum_for_get_p_k_ij(my_metric, i, my_alfa, my_beta, my_tau):
    sum = 0
    for k in range(0, len(my_metric)):
        sum += pow(my_tau[k][i], my_alfa) * pow(get_eta_ij(k, i, my_metric), my_beta)
    return sum


def goal_function(K, metric_graph):
    sum = 0
    for j, i in K:
        if metric_graph[j][i]:
            sum += metric_graph[j][i]
        else:
            return 0
    return sum


def get_eta_ij(j, i, metric_graph):
    return metric_graph[j][i]


# usunięcie duplikatu w tuplach i posortowanie względem iteracji
def removeDuplicates(lst):
    return sorted([t for t in (set(tuple(i) for i in lst))], key=lambda tup: tup[0])


# implementacja algorytmu ruletki
def roulette_selection(elements, weights):
    return choice(elements, p=weights)


# funckja sumy zamiast sigmy
def new_sum(N, fun):
    sum = 0
    for i in range(0, N):
        sum += fun(i)
    return sum


def get_adjency_matrix(G):
    G2 = np.delete(G, -1, axis=1)
    n = np.count_nonzero(G2)
    A = np.zeros((n + 2, n + 2))

    rows1 = np.where(G2[:, 0] != 0)[0]
    for i in rows1:
        A[i + 1][0] = G2[i][0]

    c1 = len(rows1) + 1
    c2 = 1
    for idx, col in enumerate(np.transpose(G2)[1:], 1):
        rows2 = np.where(col != 0)[0]
        if len(rows1):
            for j in rows1:
                tmp = c1
                for i in rows2:
                    A[c1][c2] = G2[i][idx]
                    c1 += 1
                c2 += 1
                c1 = tmp
        else:
            c2 += 1
        rows1 = rows2
        c1 += len(rows1)

    rows1 = np.where(G[:, -1] != 0)[0]
    for i in rows1:
        A[-1][c2] = G[:, -1][i]
        c2 += 1

    return A + np.transpose(A)


def get_graph_from_file(inputfile):
    with open(inputfile, 'r') as f:
        return np.array([[int(num) for num in line.split(',')] for line in f if line.strip() != ""])


def draw_graph(G, weight: bool, outputfile):
    G2 = nx.from_numpy_matrix(G, create_using=nx.DiGraph)
    f = plt.figure(3, figsize=(10, 10))
    if weight:
        layout = nx.spring_layout(G2, k=0.3 * 1 / np.sqrt(len(G2.nodes())), iterations=20)
        labels = nx.get_edge_attributes(G2, "weight")
        nx.draw(G2, pos=layout, with_labels=True, ax=f.add_subplot(111))
        nx.draw_networkx_edge_labels(G2, pos=layout, edge_labels=labels)
    else:
        nx.draw(G2, with_labels=True, ax=f.add_subplot(111))

    f.savefig(outputfile)


def plot_scutter_Je_best(Je_best, outputfile):
    plt.scatter(*zip(*Je_best))
    plt.title('Warości funkcji celu w kolejnych iteracjach')
    plt.xlabel('iteracja (t)')
    plt.ylabel('wartość funkcji celu (Je best)')
    plt.savefig(outputfile)


def plot_line_Je_best(Je_best, outputfile):
    plt.plot(*zip(*Je_best))
    plt.title('Warości funkcji celu w kolejnych iteracjach')
    plt.xlabel('iteracja (t)')
    plt.ylabel('wartość funkcji celu (Je best)')
    plt.savefig(outputfile)


def get_filename_txt(fn):
    file_name = fn
    if os.path.isfile(file_name):
        expand = 1
        while True:
            expand += 1
            new_file_name = file_name.split(".txt")[0] + str(expand) + ".txt"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
    return file_name


def get_filename_png(fn):
    file_name = fn
    if os.path.isfile(file_name):
        expand = 1
        while True:
            expand += 1
            new_file_name = file_name.split(".png")[0] + str(expand) + ".png"
            if os.path.isfile(new_file_name):
                continue
            else:
                file_name = new_file_name
                break
    return file_name


def rate_of_convergence(Je_best, t, delta):
    if t <= delta:
        return 1
    eps = np.abs((Je_best[-1][1] - Je_best[-delta][1]) / Je_best[-1][1])
    return eps


def save(tau, path, Je_best, Metric, time, dir):
    # fn_path = dir+ r'/' + 'path.txt'
    fn_tau = dir + r'/' + 'tau.txt'
    # fn_time = dir+ r'/' + 'time.txt'
    # fn_config = dir+ r'/' + 'config.txt'
    fn_result = dir + r'/' + 'result.txt'
    fn_line_je_best = dir + r'/' + 'line_je_best.png'
    fn_scutter_je_best = dir + r'/' + 'scutter_je_best.png'
    fn_weight_graph = dir + r'/' + 'weight_graph.png'

    # fn_path = get_filename_txt(fn_path)
    fn_tau = get_filename_txt(fn_tau)
    # fn_time = get_filename_txt(fn_time)
    # fn_config = get_filename_txt(fn_config)
    fn_result = get_filename_txt(fn_result)
    fn_line_je_best = get_filename_png(fn_line_je_best)
    fn_scutter_je_best = get_filename_png(fn_scutter_je_best)
    fn_weight_graph = get_filename_png(fn_weight_graph)

    # config file
    copyfile('config.py', fn_result)
    # print tablicy feromonów

    np.savetxt(fn_tau, tau, delimiter=' & ', fmt='%1.3f')
    string_to_add = r'\\'

    with open(fn_tau, 'r') as f:
        file_lines = [''.join([x.strip(), string_to_add, '\n']) for x in f.readlines()]

    with open(fn_tau, 'a') as f:
        f.writelines(file_lines)

    with open(fn_result, 'a') as f:
        f.write("\n")
        f.write("\n")
        f.write("THE RESULTS ARE: \n \n")
        f.write("The path path is: ")
        for item in path:
            f.write("%s " % item)
        f.write("\n")

    with open(fn_result, 'a') as f:
        f.write("The best path value is: ")
        f.write("%s " % Je_best[-1][1])
        f.write("\n")

    with open(fn_result, 'a') as f:
        f.write("The time execution time is: ")
        f.write("%s " % time)
        f.write("\n")
    # utils.plot_line_Je_best(Je_best[1:],fn_line_je_best)
    plot_scutter_Je_best(Je_best[1:], fn_scutter_je_best)
    # utils.draw_graph(utils.get_adjency_matrix(Metric),True,fn_weight_graph)
