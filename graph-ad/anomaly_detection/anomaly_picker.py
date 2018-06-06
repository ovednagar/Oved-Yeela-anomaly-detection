from time import strftime, gmtime
import numpy as np
import os
from graphs import Graphs
from loggers import PrintLogger, BaseLogger
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from os import path


class AnomaliyPicker:
    def __init__(self, graphs:Graphs, scores_list, database_name, logger: BaseLogger = None):
        self._database_name = database_name
        if logger:
            self._logger = logger
        else:
            self._logger = PrintLogger("default anomaly picker logger")
        self._graphs = graphs
        self._scores_list = scores_list
        self._anomalies = []
        self._anomalies_calculated = False

    def build(self):
        if not self._anomalies_calculated:
            self._anomalies_calculated = True
            self._calc()

    def anomalies_list(self):
        if not self._anomalies_calculated:
            self._anomalies_calculated = True
            self._calc()
        return self._anomalies

    def _calc(self):
        raise NotImplementedError()

    def plot_anomalies(self, file_name="anomalies"):
        raise NotImplementedError()


class ContextAnomalyPicker(AnomaliyPicker):
    def __init__(self, graphs: Graphs, scores_list, database_name, logger: BaseLogger = None, split=4, bar=0.333):
        super(ContextAnomalyPicker, self).__init__(graphs, scores_list, database_name)
        self._split = split
        self._bar = bar
        self._average_graph = []
        self._bar_graph = []

    def _calc(self):
        if len(self._scores_list) < self._split:
            self._logger.error("split number is bigger then number of graphs")
            return

        splited = []
        interval = int(len(self._scores_list) / self._split)
        for i in range(len(self._scores_list)):
            if i < interval:
                splited.append(np.average(self._scores_list[0:interval]))
            else:
                splited.append(np.average(self._scores_list[i-interval:i]))

        for avr, i in zip(splited, range(len(splited))):
            interval_bar = self._bar * avr
            self._average_graph.append(avr)
            self._bar_graph.append(interval_bar)
            if self._scores_list[i] < interval_bar:
                self._anomalies.append(i)

    def plot_anomalies(self, file_name="context_anomalies", truth=None, labels=5, info_text=None):
        plt_path = path.join("gif", self._database_name, file_name + "_" + strftime("%d:%m:%y_%H:%M:%S", gmtime()) + ".jpg")
        if "gif" not in os.listdir("."):
            os.mkdir("gif")
        if self._database_name not in os.listdir(path.join("gif")):
            os.mkdir(path.join("gif", self._database_name))

        x_axis = [x for x in range(len(self._scores_list))]
        y_axis = self._scores_list

        plt.plot(x_axis, self._average_graph)
        plt.plot(x_axis, self._bar_graph)

        plt.scatter(x_axis, y_axis, color='mediumaquamarine', marker="d", s=10)
        plt.title("parameter distribution")
        plt.xlabel("Time", fontsize=10)
        plt.ylabel("Graph", fontsize=10)

        # take 5 elements from x axis for display
        x_for_label = x_axis[::int(len(self._scores_list) / 5)]
        x_label = [self._graphs.index_to_name(x) for x in x_for_label]
        plt.xticks(x_for_label, x_label, rotation=3)
        patch = []
        if truth:
            FP = 0
            TP = 0
            FN = 0
            for x, y in zip(x_axis, y_axis):
                if x in truth and x not in self._anomalies:  # false positive
                    FP += 1
                    plt.scatter(x, y, color='red', marker="o", s=10)
                    patch.append(mpatches.Patch(label=self._graphs.index_to_name(x), color='red'))
                elif x in truth and x in self._anomalies:  # true positive
                    TP += 1
                    plt.scatter(x, y, color='green', marker="o", s=10)
                    patch.append(mpatches.Patch(label=self._graphs.index_to_name(x), color='green'))
                elif x not in truth and x in self._anomalies:  # false negative
                    FN += 1
                    plt.scatter(x, y, color='black', marker="o", s=10)
                    patch.append(mpatches.Patch(label=self._graphs.index_to_name(x), color='black'))
                    # the is true negative
                plt.legend(handles=patch, fontsize='small', loc=2)
            info_text += "FP:" + str(FP) + "  ||  TP:" + str(TP) + "  ||  FN:" + str(FN)
            plt.text(max(x_axis) * 0.7, max(y_axis) * 0.65, info_text)
        else:
            for x, y in zip(x_axis, y_axis):
                if x in self._anomalies:  # predict anomaly
                    plt.scatter(x, y, color='green', marker="o", s=10)
                    patch.append(mpatches.Patch(label=self._graphs.index_to_name(x), color='green'))
                plt.legend(handles=patch, fontsize='small', loc=2)
            plt.text(max(x_axis) * 0.7, max(y_axis) * 0.7, info_text)

        plt.savefig(plt_path)
        plt.clf()
        plt.close()
