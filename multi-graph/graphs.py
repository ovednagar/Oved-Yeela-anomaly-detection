from feature_meta import NODE_FEATURES
from multi_graph_features import MultiGraphFeatures
from loggers import BaseLogger, PrintLogger
import pickle
import os


"""
 if graph is loaded from files for the first time
    - each graph should be represented by a different file
    - all files should be in the same directory
    - the file for a graph will contain a list of edges 
         node node weight(weight is optional)
"""


class Graphs:
    def __init__(self, database_name, logger: BaseLogger=None, features_meta=None,
                 directed=False, files_path=None, date_format=None, largest_cc=False):
        self._features_meta = NODE_FEATURES if features_meta is None else features_meta
        self._largest_cc = largest_cc
        self._date_format = date_format
        self._directed = directed
        self._database_name = database_name + "_directed:" + str(directed) + "_lcc:" + str(largest_cc)
        self._path = os.path.join('data', self._database_name)
        if logger:
            self._logger = logger
        else:
            self._logger = PrintLogger("default graphs logger")
        self._files_path = files_path  # location of graphs as files

        # make directories to save features data (as pickles)
        if "data" not in os.listdir("."):
            os.mkdir("data")
        if self._database_name not in os.listdir("data/"):
            os.mkdir(self._path)
        self._logger.debug("graphs initialized")

    def get_feature_meta(self):
        return self._multi_graph.get_feature_meta()

    def is_directed(self):
        return self._directed

    def _is_loaded(self):
        return True if os.path.exists(self._get_pickle_path()) else False

    def _load_graphs(self):
        self._logger.debug("load multi-graph - start")
        self._multi_graph = pickle.load(open(self._get_pickle_path(), "rb"))
        self._logger.debug("pickle loaded")
        self._logger.debug("load multi-graph - end")

    def _dump(self, redump=False):
        if self._is_loaded() and not redump:
            self._logger.debug("multi-graph is already loaded")
            return
        log = self._multi_graph._logger
        key_func = self._multi_graph._key_func
        self._multi_graph._logger = None
        self._multi_graph._key_func = None
        pickle.dump(self._multi_graph, open(self._get_pickle_path(), "wb"))
        self._multi_graph._logger = log
        self._multi_graph._key_func = key_func
        self._logger.debug("multi-graph dumped")

    def _get_pickle_path(self):
        return os.path.join(self._path, self._database_name + ".pkl")

    def build(self, force_build: bool = False, pick_ftr=False, force_rebuild_ftr=False):
        if force_build or not os.path.exists(self._get_pickle_path()):
            self._logger.debug("build multi-graph - start")
            self._multi_graph = MultiGraphFeatures(self._database_name, self._logger, features_meta=self._features_meta,
                                                   directed=self._directed, files_path=self._files_path
                                                   , pkl_dir=self._path, date_format=self._date_format)
            if not self._multi_graph.build():
                return
            self._multi_graph.build_features(largest_cc=self._largest_cc)
            self._dump()
            self._logger.debug("build multi-graph - end")
        else:
            self._load_graphs()
            self._multi_graph._logger = self._logger
            self._multi_graph._pkl_dir = self._path
            if pick_ftr or force_rebuild_ftr:
                self._multi_graph._features_meta = self._features_meta
                self._multi_graph.build_features(pick_ftr=pick_ftr, force_rebuild=force_rebuild_ftr, largest_cc=self._largest_cc)
                self._dump(redump=True)

    # Adapter for multi graph
    def get_subgraph(self, graph_name):
        return self._multi_graph.subgraph_by_name(graph_name)

    def subgraph_by_name(self, graph_name: str):
        return self._multi_graph.subgraph_by_name(graph_name)

    def subgraph_by_index(self, index: int):
        return self._multi_graph.subgraph_by_index(index)

    def combined_graph_by_names(self, names_list=None, combine_all=False):
        return self._multi_graph.combined_graph_by_names(names_list, combine_all)

    def combined_graph_by_indexes(self, index_list=None, combine_all=False):
        return self.combined_graph_by_indexes(index_list, combine_all)

    def is_graph(self, graph_name):
        return self._multi_graph.is_graph(graph_name)

    def index_to_name(self, index):
        return self._multi_graph.index_to_name(index)

    def name_to_index(self, graph_name):
        return self._multi_graph.name_to_index(graph_name)

    def features_matrix_by_index(self, graph_start=0, graph_end=0, for_all=False):
        return self._multi_graph.features_matrix_by_indexes(graph_start, graph_end, for_all)

    def features_matrix_by_name(self, graph_start=0, graph_end=0, for_all=False):
        return self._multi_graph.features_matrix_by_names(graph_start, graph_end, for_all)

    def features_matrix(self, graph):
        return self._multi_graph.feature_matrix(graph)

    def nodes_for_graph(self, graph):
        return self._multi_graph.nodes_for_graph(graph)

    def nodes_count_list(self):
        return self._multi_graph.nodes_count_list()

    def subgraphs(self, start_id=None, end_id=None):
        for gid in self._multi_graph._list_id[start_id: end_id]:
            yield self.subgraph_by_name(gid)

    def number_of_graphs(self):
        return self._multi_graph.number_of_graphs()

    def graph_names(self, start_id=None, end_id=None):
        for gid in self._multi_graph._list_id[start_id: end_id]:
            yield gid


def test_graph():
    logger = PrintLogger("Oved's logger")
    path = "test_graphs"
    graphs = Graphs("test - Debug", logger=logger, files_path=path)
    graphs.build()
    G_1 = graphs.get_subgraph("time_1")
    G_2 = graphs.get_subgraph("time_2")
    G_3 = graphs.get_subgraph("time_3")

    stop = 0


if __name__ == "__main__":
    pass
