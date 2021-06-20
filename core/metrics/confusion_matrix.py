import json

import numpy as np


def to_categorical(targets):
    return np.argmax(targets, axis=1)


class ClassificationReport:
    def __init__(self, targets, preds):
        if targets.shape[1] > 1:
            targets = to_categorical(targets)
            preds = to_categorical(preds)

        self.labels = np.unique(targets)
        self.report = {key: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for key in self.labels}
        for label in self.labels:
            self.report[label]["tp"] = np.sum(np.logical_and(preds == label, targets == label))
            self.report[label]["tn"] = np.sum(np.logical_and(preds != label, targets != label))
            self.report[label]["fp"] = np.sum(np.logical_and(preds == label, targets != label))
            self.report[label]["fn"] = np.sum(np.logical_and(preds != label, targets == label))

    def __repr__(self):
        return self.report

    def __str__(self):
        return str(self.report)

    def __call__(self):
        return self.report
