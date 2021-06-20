import importlib

import numpy as np

from .callback import Callback
from ..metrics import Metric, ClassificationReport
from ..utils import merge_dicts


class MetricsCallback(Callback):
    def __init__(self, selected_metrics, on_epoch_functions=None):
        if on_epoch_functions is None:
            on_epoch_functions = []

        self.model = None
        self.selected_metrics = selected_metrics
        self.on_epoch_functions = on_epoch_functions

    def set_modelo(self, model):
        self.model = model

    def init(self, **kwargs):
        for metric_name in self.selected_metrics:
            targets = kwargs["targets"]
            scores = kwargs["scores"] if kwargs["scores"].ndim == 2 else np.expand_dims(kwargs["scores"], axis=1)
            report = ClassificationReport(targets, scores)
            if metric_name not in self.model.computed_metrics.keys():
                metric_fn = getattr(importlib.import_module("core.metrics"), str(metric_name))(report)
                self.model.computed_metrics[metric_name] = {"function": metric_fn, "value_per_epoch": {}}
            else:
                self.model.computed_metrics[metric_name]["function"].report = report

        if not self.are_metrics([metric["function"] for metric in self.model.computed_metrics.values()]):
            raise ValueError(
                "Alguna de las métricas proporcionadas no procede de la clase Metrics del módulo utils")

    def on_epoch_end(self, epoch, **kwargs):
        function_return = {"coste": [self.model.cost[epoch]]}
        for metric in self.model.computed_metrics.keys():
            function_return[metric] = self.model.computed_metrics[metric]["function"].compute()
            merge_dicts(self.model.computed_metrics[metric]["value_per_epoch"], function_return[metric])

        for func in self.on_epoch_functions:
            func(function_return)

    def on_training_end(self):
        pass

    @staticmethod
    def are_metrics(metrics):
        for metric in metrics:
            if not issubclass(metric.__class__, Metric):
                return False
        return True
