from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    def __init__(self, report=None):
        self.report = report
        self.values = {str(label): [] for label in report.labels} if report else None

    @property
    def report(self):
        return self._report

    @report.setter
    def report(self, report):
        self._report = report
        self.values = {str(label): [] for label in report.labels} if report else None

    @abstractmethod
    def compute(self):
        pass

    def __str__(self):
        return self.name


class Accuracy(Metric):
    name = 'Accuracy'

    def compute(self):
        for label, conf in self.report().items():
            total_preds = np.sum(list(conf.values()))
            self.values[str(label)].append((conf["tp"] + conf["tn"]) / total_preds)
        self.values["macro"] = [np.mean(list(self.values.values()))]
        return self.values


class Recall(Metric):
    name = 'Recall'

    def compute(self):
        for label, conf in self.report().items():
            self.values[str(label)].append(conf["tp"] / (conf["tp"] + conf["fn"]))
        self.values["macro"] = [np.mean(list(self.values.values()))]
        return self.values


class Precision(Metric):
    name = 'Precision'

    def compute(self):
        for label, conf in self.report().items():
            self.values[str(label)].append(conf["tp"] / (conf["tp"] + conf["fp"]))
        self.values["macro"] = [np.mean(list(self.values.values()))]
        return self.values


class FbScore(Metric):
    name = 'FbScore'

    def __init__(self, report, beta=1):
        super().__init__(report)
        self.beta = beta

    def compute(self):
        for label, conf in self.report().items():
            self.values[str(label)].append((1 + self.beta ** 2) * conf["tp"] / (
                    (1 + np.power(self.beta, 2)) * conf["tp"] + np.power(self.beta, 2) * conf["fn"] + conf["fp"]))
        self.values["macro"] = [np.mean(list(self.values.values()))]
        return self.values
