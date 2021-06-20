import numpy as np

from core.metrics import ClassificationReport, Accuracy, Recall, Precision, FbScore

targets = np.array(np.expand_dims([1, 0, 1, 1], axis=1))
preds = np.array(np.expand_dims([1, 0, 0, 1], axis=1))

conf = ClassificationReport(targets, preds)

# print("Accuracy: " + str(Accuracy(conf).compute()))
# print("Recall: " + str(Recall(conf).compute()))
# print("Precision: " + str(Precision(conf).compute()))
# print("F1 score: " + str(FbScore(conf).compute()))

print(Accuracy(conf))

