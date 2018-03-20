import numpy as np
import numba as nb
from numba import jitclass
from numba import jit

cluster_spec = [('names', nb.int32[:]), ('correl_matrix', nb.float32[:, :]), ('n', nb.int32), ('c', nb.float32),
                ('log_likelihood', nb.float32)]
@jitclass(cluster_spec)
class cluster:
    def __init__(self, names, correl_matrix):
        self.names = names
        self.n = len(names)
        self.c = 0.
        for name in self.names:
            for name2 in self.names:
                self.c += correl_matrix[name, name2]
        n = len(self.names)
        c = self.c
        if n > 1:
            self.log_likelihood = np.log(n / c) + (n - 1) * np.log((n * n - n) / (n * n - c))
        else:
            self.log_likelihood = 0

    @property
    def average_correl(self):
        if self.n == 1: return 1
        else: return (self.c - self.n) / (self.n * (self.n - 1))

@jit
def merge_clusters(cluster1, cluster2, C):
    names = np.concatenate((cluster1.names, cluster2.names))
    return cluster(names, C)


# cluster_type = nb.deferred_type()
# cluster_type.define(cluster.class_type.instance_type)
# cluster_net_spec = [('names', nb.int32[:]), ('correl_matrix', nb.float32[:, :]), ('clusters', cluster_type[:]),
#                     ('log_likelihood', nb.float32)]
# @jitclass(cluster_net_spec)
class cluster_net:
    def __init__(self, names, correl_matrix):
        self.correl_matrix = correl_matrix
        self.names = names
        self.clusters = np.zeros(len(names), dtype=np.object)
        for name in names:
            temp = name + np.zeros(1, dtype=np.int32)
            self.clusters[name] = cluster(temp, correl_matrix)
        self.log_likelihood = 0
        for clust in self.clusters:
            self.log_likelihood += clust.log_likelihood

    @jit
    def merge_clusters(self, cluster1, cluster2):
        index = np.argwhere(self.clusters == cluster1)
        self.clusters = np.delete(self.clusters, index)
        index = np.argwhere(self.clusters == cluster2)
        self.clusters = np.delete(self.clusters, index)
        self.clusters = np.append(self.clusters, [merge_clusters(cluster1, cluster2, self.correl_matrix)])
        self.log_likelihood = 0
        for cluster in self.clusters:
            self.log_likelihood += cluster.log_likelihood

    @jit
    def find_best_merge(self, only_likelihood_improve=False):
        C = self.correl_matrix
        best_improve = -np.inf
        best_pair = (np.nan, np.nan)
        clusts = self.clusters
        while len(clusts) > 1:
            cluster1 = clusts[0]
            clusts = clusts[1:]
            for cluster2 in clusts:
                if only_likelihood_improve:
                    improve = merge_clusters(cluster1, cluster2, C).log_likelihood \
                              - cluster1.log_likelihood - cluster2.log_likelihood
                else:
                    improve = merge_clusters(cluster1, cluster2, C).log_likelihood \
                              - max([cluster1.log_likelihood, cluster2.log_likelihood])
                if improve > best_improve:
                    best_improve = improve
                    best_pair = (cluster1, cluster2)
        return best_improve, best_pair

    @jit
    def best_merge(self, only_likelihood_improve=False):
        improve, pair = self.find_best_merge(only_likelihood_improve)
        cluster1, cluster2 = pair
        if improve > 0:
            self.merge_clusters(cluster1, cluster2)
            return True
        else:
            return False

    @jit
    def successive_merge(self, N, only_likelihood_improve=False, print_state=True):
        i = 0
        while i < N and self.best_merge(only_likelihood_improve):
            i = i + 1
        if print_state: print('Total number of merges:', i)

    @jit
    def get_cluster(self, name):
        if name not in self.names:
            raise Exception(str(name) + ' not in names')
        else:
            i = 0
            for cluster in self.clusters:
                i = i + 1
                if name in cluster.names:
                    return i
