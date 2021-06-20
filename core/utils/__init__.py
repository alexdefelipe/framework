import numpy as np


def round_scores(model, scores):
    rounded_scores = np.array([int(np.round(score)) if model.n_classes is 2 and model.multiclass is False
                               else np.argmax(score) for score in scores])
    return rounded_scores


def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1.keys():
            dict1[key].extend(value)
        else:
            dict1[key] = value
