import numpy as np

class cluster:
    def __init__(self, names, correl_matrix):
        names = list(set(names))
        self.names = names
        self.n = len(names)
        self.c = np.sum(correl_matrix.loc[names, names].values)
        self.log_likelihood = self.log_likelihood()

    def log_likelihood(self):
        n = len(self.names)
        c = self.c
        if n > 1:
            return np.log(n / c) + (n - 1) * np.log((n * n - n) / (n * n - c))
        else:
            return 0


def merge_clusters(cluster1, cluster2, C):
    names = list(set().union(cluster1.names, cluster2.names))
    return cluster(names, C)


class cluster_net:
    def __init__(self, names, correl_matrix):
        names = list(set(names))
        self.correl_matrix = correl_matrix
        self.names = names
        self.clusters = list([cluster([name], correl_matrix) for name in names])
        self.log_likelihood = self.log_likelihood()

    def log_likelihood(self):
        return np.sum([cluster.log_likelihood for cluster in self.clusters])

    def merge_clusters(self, cluster1, cluster2):
        self.clusters.remove(cluster1)
        self.clusters.remove(cluster2)
        self.clusters.append(merge_clusters(cluster1, cluster2, self.correl_matrix))

    def find_best_merge(self, only_likelihood_improve=False):
        C = self.correl_matrix
        best_improve = -np.inf
        best_pair = (np.nan, np.nan)
        clusters = list(self.clusters)
        while len(clusters) > 1:
            cluster1 = clusters.pop(0)
            for cluster2 in clusters:
                if only_likelihood_improve:
                    improve = merge_clusters(cluster1, cluster2,
                                             C).log_likelihood - cluster1.log_likelihood - cluster2.log_likelihood
                else:
                    improve = merge_clusters(cluster1, cluster2, C).log_likelihood - max(
                        [cluster1.log_likelihood, cluster2.log_likelihood])
                if improve > best_improve:
                    best_improve = improve
                    best_pair = (cluster1, cluster2)
        return best_improve, best_pair

    def best_merge(self, only_likelihood_improve=False):
        improve, pair = self.find_best_merge(only_likelihood_improve)
        cluster1, cluster2 = pair
        if improve > 0:
            self.merge_clusters(cluster1, cluster2)
            return True
        else:
            return False

    def successive_merge(self, N, only_likelihood_improve=False):
        i = 0
        while i < N and self.best_merge(only_likelihood_improve):
            i = i + 1
        print('Total number of merges:', i)

    def get_cluster(self, name):
        if name not in self.names:
            raise Exception(name + ' not in names')
        else:
            i = 0
            for cluster in self.clusters:
                i = i + 1
                if name in cluster.names:
                    return i
