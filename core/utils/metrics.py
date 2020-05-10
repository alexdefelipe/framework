from enum import Enum


class Metrics(Enum):
    AUC, ACCURACY, RECALL, F1_SCORE, LOSS = range(5)
