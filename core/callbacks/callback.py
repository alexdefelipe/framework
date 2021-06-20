from abc import ABC, abstractmethod


class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, epoch, **kwargs):
        pass

    @abstractmethod
    def on_training_end(self):
        pass

    @abstractmethod
    def set_modelo(self, modelo):
        pass

    @abstractmethod
    def init(self, **kwargs):
        pass
