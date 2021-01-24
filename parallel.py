if __name__ == "__main__":
    from pyspark import SparkContext

    sc = SparkContext("local", "MySparkApp")
    
    import copy
    import time
    import numpy as np
    import utils
    from config import alfa, beta, rho, C, M, tau_0, fn_input_metric, N, delta, ratio
    from utils import new_sum, roulette_selection

    Metric = utils.get_graph_from_file(fn_input_metric)
    # liczba kolumn
    D = len(Metric[0])
    # liczba wierszy
    R = len(Metric)
    # tau inizjalizacja
    tau = sc.broadcast(np.array((R, D)))
    tau[...] = tau_0
    # tau k inizjalizacja
    tau_all = sc.broadcast(np.array((M, R, D)))
    tau_all[...] = tau_0
    # tau best inizjalizacja
    tau_best = sc.broadcast(np.array((R, D)))
    # Wartość Je_best best przechowuje [t,Je_best] t kolejna iteracja a Je_best to wartość funkcji celu
    Je_best = 0
    Je_best_t = sc.broadcast([(0, 0)])

    # definicja funkcji celu na chwilę obecną równomiernie rozprowadza feromony na węzły które należą do najlepszej ścieżki


def goal_function(K):
    sum = 0
    for j, i in K:
        if Metric[j][i]:
            sum += Metric[j][i]
        else:
            return 0
    # rówżnomiernie rozprowadzamy wartość funkcji celu na poszczególny węzeł
    return sum


# TODO do wymyślenia funkcja heurystyczna na chwilę obecną zwraca wartość metryki
def get_eta_ij(t, j, i):
    return Metric[j][i]


def set_delta_tau_k_ij(t, k, K, Je):
    for j, i in K:
        if Metric[j][i] and Je:
            tau_all[k][j][i] = 1 / Je
        else:
            tau_all[k][j][i] = 0


def set_delta_tau_best_ij(t, K, Je):
    global Je_best
    if Je_best == 0 or Je_best > Je:
        Je_best = Je
        tau_best[...] = .0
        for j, i in K:
            tau_best[j][i] = Je


# TODO pytanie czy zapamiętujemy wartość tablicy feromonów po każdej iteracji
def set_tau_ij(t, j, i):
    def tau_func(k):
        return tau_all[k][j][i]

    tau[j][i] = (1 - rho) * tau[j][i] + new_sum(M, tau_func) + rho * tau_best[j][i]


# Prawdopodobieństwo wybrania węzła j będąc na węźle i
def get_p_k_ij(t, k, j, i):
    def tau_func(x):
        return pow(tau[x][i], alfa) * pow(get_eta_ij(t, x, i), beta)

    return pow(tau[j][i], alfa) * pow(get_eta_ij(t, j, i), beta) / new_sum(R, tau_func)


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
    elements = [e for e in range(0, R)]
    # ilość cykli
    t = 1
    while utils.rate_of_convergence(Je_best_t, t, delta) > ratio and t < C:
        # ilość mrówek
        rddata = sc.parallelize(N)
        rddata = rddata.map(ant_function)
        return rddata.take()


def ant_function():
    for k in p.range(0, M):
        # losowanie ścieżek
        K = []
        for i in range(0, D):
            weights = []
            with p.lock:
                for j in range(0, R):
                    weights.append(get_p_k_ij(t, k, j, i))
            j = roulette_selection(elements, weights)
            K.append((j, i))

        Je = goal_function(K)
        # Pozostawienie feromonu po przejściu pierwszej ścieżki
        set_delta_tau_k_ij(t, k, K, Je)
        # sprawdzamy czy najlepsza ścieżka do tej pory
        with p.lock:
            set_delta_tau_best_ij(t, K, Je)
    # ustawiamy tablice feromonuów
    with p.lock:
        for i in range(0, D):
            for j in range(0, R):
                set_tau_ij(t, j, i)
    Je_best_t.append((t, Je_best))
    t += 1


if __name__ == "__main__":
    t = time.process_time()
    run_aco_algorithm()
    elapsed_time = time.process_time() - t
    print("Time")
    print(elapsed_time)
    print("Best path value")
    print(Je_best_t[-1][1])
    print("Best path value2")
    print(Je_best_t[-delta][1])
    print("Liczba iteracji")
    print(Je_best_t[-1][0])
    utils.save(get_tau(), get_path(), Je_best_t, Metric, elapsed_time, 'out_parallel')
