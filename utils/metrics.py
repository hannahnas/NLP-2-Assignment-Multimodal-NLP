import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)
LOGGER = logging.getLogger('MetricLogger')


def standard_metrics(probs, labels, *args, **kwargs):
    if len(probs.shape) == 1 and torch.all(torch.logical_or(labels == 0, labels == 1)):
        return standard_metrics_binary(probs, labels, *args, **kwargs)
    else:
        raise ValueError("[!] ERROR: labels are not binary!!")


def standard_metrics_binary(probs, labels, threshold=0.5, add_aucroc=True, add_optimal_acc=False, **kwargs):
    """
    Given predicted probabilities and labels, returns the standard metrics of accuracy, recall, precision, F1 score and AUCROC.
    The threshold, above which data points are considered to be classified as class 1, can also be adjusted.
    Returned values are floats (no tensors) in the dictionary 'metrics'.
    Probabilities and labels are expected to be pytorch tensors.
    """
    preds = torch.where(probs < threshold, 0, 1)

    # true positives
    correct = torch.count_nonzero(preds.eq(labels))
    acc = correct / len(probs)

    # Check for numerical stability, otherwise we divide by zero if all labels are zero
    if torch.count_nonzero(labels).item() == 0:
        recall = 0
    else:
        recall = correct / torch.count_nonzero(labels).item()

    precision = correct / torch.count_nonzero(preds)
    f1 = 2 * (precision * recall) / (precision + recall)
    aucroc_score = aucroc(labels, preds)
    metrics = {'accuracy': acc,
               'recall': recall,
               'precision': precision,
               'F1': f1,
               'AUCROC': aucroc_score}
    return metrics


# OPTIONAL:  you can also optimize the cut-off threshold for the binary classification task (default=0.5)
def find_optimal_threshold(probs, labels, metric="accuracy"):
    """
    Given predicted probabilities and labels, returns the optimal threshold to use for the binary classification.
    It is conditioned on a metric ("accuracy", "F1", ...). Probabilities and labels are expected to be pytorch tensors.
    """
    # YOUR CODE HERE:  write code to find the best_threshold from a range of tested ones optimizing for the given metric
    return best_threshold


def aucroc(probs, labels):
    """
    Given predicted probabilities and labels, returns the AUCROC score used in the Facebook Meme Challenge.
    Inputs are expected to be pytorch tensors (can be cuda or cpu)
    """
    aucroc_score = roc_auc_score(labels, probs)
    return aucroc_score


if __name__ == '__main__':
    num_classes = 2
    probs = torch.randn(size=(1000,num_classes))
    probs = F.softmax(probs, dim=-1)
    labels = torch.multinomial(probs, num_samples=1).squeeze() * 0
    print("Metrics", standard_metrics(probs[:, 1], labels))