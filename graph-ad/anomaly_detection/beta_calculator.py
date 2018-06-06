from sklearn import linear_model
from graphs import Graphs
from loggers import BaseLogger, PrintLogger
import numpy as np


class BetaCalculator:
    def __init__(self, graphs: Graphs, feature_pairs, logger: BaseLogger=None):
        if logger:
            self._logger = logger
        else:
            self._logger = PrintLogger("default graphs logger")
        self._graphs = graphs
        self._ftr_pairs = feature_pairs
        self._beta_matrix = np.zeros((self._graphs.number_of_graphs(), len(feature_pairs)))
        self._build()

    def _build(self):
        graph_index = 0
        for g_id in self._graphs.graph_names():
            self._logger.debug("calculating beta vec for:\t" + g_id)
            self._beta_matrix[graph_index, :] = self._calc_beta(g_id)
            graph_index += 1

    def _calc_beta(self, gid):
        raise NotImplementedError()

    def beta_matrix(self):
        return self._beta_matrix

    def to_file(self, file_name):
        out_file = open(file_name, "rw")
        for i in range(self._graphs.number_of_graphs()):
            out_file.write(self._graphs.index_to_name(i))  # graph_name
            for j in range(len(self._ftr_pairs)):
                out_file.write(str(self._beta_matrix[i][j]))  # beta_vector
            out_file.write("\n")
        out_file.close()


class LinearContext(BetaCalculator):
    def __init__(self, graphs: Graphs, feature_pairs, logger: BaseLogger = None, split=1):
        self._interval = int(graphs.number_of_graphs() / split)
        self._all_features = graphs.features_matrix_by_index(for_all=True)
        self._nodes_for_graph = graphs.nodes_count_list()
        self._all_ftr_graph_index = []
        for i in range(len(self._nodes_for_graph) + 1):
            self._all_ftr_graph_index.append(np.sum(self._nodes_for_graph[0:i]))
        super(LinearContext, self).__init__(graphs, feature_pairs, logger=logger)

    def _linear_regression(self, x_np, y_np):
        """
        :param x_np: numpy array for one feature
        :param y_np:
        :return: regression coefficient
        """
        return linear_model.LinearRegression().fit(np.transpose(x_np), np.transpose(y_np)).coef_[0][0]

    def _calc_beta(self, gid):
        beta_vec = []
        # get features matrix for interval
        g_index = self._graphs.name_to_index(gid)

        if g_index < self._interval:
            context_matrix = self._all_features[0: int(self._all_ftr_graph_index[self._interval]), :]
        else:
            context_matrix = self._all_features[int(self._all_ftr_graph_index[g_index - self._interval]):
                                                int(self._all_ftr_graph_index[g_index]), :]
        # get features matrix only for current graph
        g_matrix = self._graphs.features_matrix(gid)

        for i, j in self._ftr_pairs:
            beta_vec.append(np.mean(g_matrix[:, j] -
                            linear_model.LinearRegression().fit(np.transpose(context_matrix[:, i].T),
                                                                np.transpose(context_matrix[:, j].T)).coef_[0][0] *
                            g_matrix[:, i]))
        return np.asarray(beta_vec)


class LinearRegBetaCalculator(BetaCalculator):
    def __init__(self, graphs: Graphs, feature_pairs, single_c=True):
        self.single_c = single_c
        self.single_c_calculated = False
        self.single_c_regression = {}
        super(LinearRegBetaCalculator, self).__init__(graphs, feature_pairs)

    def _calc_single(self, feature_i, feature_j):
        # if we calculate a single c over all the graphs
        f_key = str(feature_i) + "," + str(feature_j)
        if f_key not in self.single_c_regression:
            self._ftr_matrix = self._graphs.features_matrix_by_name(for_all=True)
            self.single_c_regression[f_key] = self._linear_regression(self._ftr_matrix[:, feature_i].T,
                                                                  self._ftr_matrix[:, feature_j].T)
        return self.single_c_regression[f_key]

    def _calc_multi(self, feature_vec_i, feature_vec_j):
        # here we calculate c for each graph separately
        return self._linear_regression(feature_vec_i, feature_vec_j)

    def _linear_regression(self, x_np, y_np):
        """
        :param x_np: numpy array for one feature
        :param y_np:
        :return: regression coefficient
        """
        x = x_np.tolist()
        y = y_np.tolist()
        regression = linear_model.LinearRegression()
        regression.fit(np.transpose(np.matrix(x)), np.transpose(np.matrix(y)))
        return regression.coef_[0][0]

    def _calc_beta(self, gid):
        beta_vec = []
        for i, j in self._ftr_pairs:
            ftr_matrix = self._graphs.features_matrix(gid)
            feature_i = ftr_matrix[:, i].T
            feature_j = ftr_matrix[:, j].T

            # if we calculate a single c over all the graphs
            if self.single_c:
                coef_ij = self._calc_single(i, j)
            else:
                coef_ij = self._calc_multi(feature_i, feature_j)

            # calculate b_ijk which is defined as the mean on the b for all the vertices of a graph
            # b_ijk_vec is a vector of b_ijk for all the vertices in the graph
            b_ijk_vec = feature_j - coef_ij * feature_i
            b_ijk = np.mean(b_ijk_vec)
            beta_vec.append(b_ijk)
        return np.asarray(beta_vec)
