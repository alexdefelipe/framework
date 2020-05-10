from .callback import Callback
from ..utils import Metrics


class MetricsCallback(Callback):
    def __init__(self, metrics=[Metrics.LOSS], on_epoch_functions=[]):
        self.model = None
        self.on_epoch_functions = on_epoch_functions
        if self.are_metrics(metrics):
            self.metrics = {k: [] for k in metrics}
        else:
            raise ValueError(
                "Alguna de las métricas proporcionadas no procede de la clase Metrics del módulo utils")

    def set_modelo(self, model):
        self.model = model

    def on_epoch_end(self, epoch):
        function_return = {}
        for metric in self.metrics.keys():
            if metric == Metrics.LOSS:
                self.metrics[metric] = self.model.cost[epoch]
                function_return[metric.name] = self.model.cost[epoch]
        for func in self.on_epoch_functions:
            func(function_return)

    def on_training_end(self):
        pass

    @staticmethod
    def are_metrics(metrics):
        for metric in metrics:
            if not isinstance(metric, Metrics):
                return False
        return True
