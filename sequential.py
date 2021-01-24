import copy
import time

import numpy as np
from typing import List

import utils
from config import alfa, beta, rho, C, M, tau_0, fn_input_metric, delta, ratio
from utils import roulette_selection, goal_function, get_eta_ij, new_sum_for_set_tau_ij, new_sum_for_get_p_k_ij, \
    get_p_k_ij, AntResult

Metric = utils.get_graph_from_file(fn_input_metric)
# tau inizjalizacja
tau = np.zeros((len(Metric), len(Metric[0])))
tau[...] = tau_0
# tau k inizjalizacja
tau_all = np.zeros((M, len(Metric), len(Metric[0])))
tau_all[...] = tau_0
# tau best inizjalizacja
tau_best = np.zeros((len(Metric), len(Metric[0])))
# Wartość Je_best best przechowuje [t,Je_best] t kolejna iteracja a Je_best to wartość funkcji celu
Je_best = 0
Je_best_t = [(0, 0)]


# TODO pytanie czy zapamiętujemy wartość tablicy feromonów po każdej iteracji
def set_tau_ij(j, i):
    tau[j][i] = (1 - rho) * tau[j][i] + new_sum_for_set_tau_ij(M, tau_all, i, j) + rho * tau_best[j][i]


def get_tau():
    a = copy.deepcopy(tau)
    for i, row in enumerate(tau, 0):
        for j, col in enumerate(row, 0):
            if Metric[i][j] == 0:
                a[i][j] = 0
    return np.round(a, decimals=2)


def get_path():
    G = copy.deepcopy(Metric)
    c = 1
    for j, col in enumerate(np.transpose(G), 0):
        for i, row in enumerate(col, 0):
            if G[i][j] != 0:
                G[i][j] = c
                c += 1
    path = [0]
    for j, col in enumerate(np.transpose(G)):
        path.append(G[tau.argmax(0)[j]][j])
    n = np.count_nonzero(Metric)
    path.append(n + 1)
    return path


def run_aco_algorithm():
    elements = [e for e in range(0, len(Metric))]
    # ilość cykli
    t = 1
    while utils.rate_of_convergence(Je_best_t, t, delta) > ratio and t < C:
        results: List[AntResult] = []
        # ilość mrówek
        for k in range(0, M):
            results.append(one_ant(k, elements))
        print(t)

        for k in range(0,M):
            ant: AntResult = results[k]
            # Pozostawienie feromonu po przejściu pierwszej ścieżki
            set_delta_tau_k_ij(ant.ant_id, ant.my_k, ant.my_je)
            # sprawdzamy czy najlepsza ścieżka do tej pory
            set_delta_tau_best_ij(ant.my_k, ant.my_je)
        # ustawiamy tablice feromonuów
        for i in range(0, len(Metric[0])):
            for j in range(0, len(Metric)):
                set_tau_ij(j, i)
        Je_best_t.append((t, Je_best))
        t += 1


def one_ant(ant_id, elements) -> AntResult:
    # losowanie ścieżek
    K = []
    for i in range(0, len(Metric[0])):
        weights = []
        for j in range(0, len(Metric)):
            weights.append(get_p_k_ij(j, i, tau, alfa, Metric, beta))
        j = roulette_selection(elements, weights)
        K.append((j, i))
    Je = goal_function(K, Metric)

    return AntResult(K, Je, ant_id)


def set_delta_tau_k_ij(k, K, Je):
    for j, i in K:
        if Metric[j][i] and Je:
            tau_all[k][j][i] = 1 / Je
        else:
            tau_all[k][j][i] = 0


def set_delta_tau_best_ij(K, Je):
    global Je_best
    if Je_best == 0 or Je_best > Je:
        Je_best = Je
        tau_best[...] = .0
        for j, i in K:
            tau_best[j][i] = Je


if __name__ == "__main__":
    t = time.process_time()
    run_aco_algorithm()
    elapsed_time = time.process_time() - t
    print("Time")
    print(elapsed_time)
    print("Best path value")
    print(Je_best_t[-1][1])
    print("Liczba iteracji")
    print(Je_best_t[-1][0])
    utils.save(get_tau(), get_path(), Je_best_t, Metric, elapsed_time, 'out_seq')
    utils.draw_graph(utils.get_adjency_matrix(Metric), True, 'graph5x5.png')
