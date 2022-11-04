import torch
import numpy as np

th = 0.5

def true_positives(target, output):

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > th
    target_ = target > th

    TP = np.sum((target_ == 1) & (output_ == 1))

    return TP


def false_positives(target, output):

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > th
    target_ = target > th

    FP = np.sum((target_ == 0) & (output_ == 1))

    return FP


def true_negatives(target, output):

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > th
    target_ = target > th

    TN = np.sum((target_ == 0) & (output_ == 0))

    return TN


def false_negatives(target, output):

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > th
    target_ = target > th

    FN = np.sum((target_ == 1) & (output_ == 0))

    return FN


# False Positive Rate - Fall-Out
def fpr(fp, tn):

    out = fp / (fp + tn)

    return out


def precision(tp, fp):

    out = tp / (tp + fp)

    return out


# True Positive rate - Recall
def tpr(tp, fn):

    out = tp / (tp + fn)

    return out


def dsc(tp, fp, fn):

    dice = (2 * tp) / (2 * tp + fp + fn)

    return dice


def calculate_all_measures(y_true, y_pred, epsilon=1e-7):

    global output, target

    if torch.is_tensor(y_pred):
        output = y_pred.data.cpu().numpy()
    if torch.is_tensor(y_true):
        target = y_true.data.cpu().numpy()

    output_ = output > th
    target_ = target > th

    tp = np.sum((target_ == 1) & (output_ == 1))
    tn = np.sum((target_ == 0) & (output_ == 0))
    fp = np.sum((target_ == 0) & (output_ == 1))
    fn = np.sum((target_ == 1) & (output_ == 0))

    precision_v = tp / (tp + fp)
    recall_v = tp / (tp + fn)
    fallout_v = fp / (fp + tn)

    f1 = 2 * (precision_v * recall_v) / (precision_v + recall_v)

    return tp, fp, tn, fn, precision_v, recall_v, fallout_v, f1

def confusion(truth, prediction):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    th = torch.mean(truth).item()
    output_ = (prediction > th).float()
    target_ = (truth > th).float()

    confusion_vector = output_ // target_
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()

    try:
        precision_v = tp / (tp + fp)
    except ZeroDivisionError:
        precision_v = 0

    try:
        recall_v = tp / (tp + fn)
    except ZeroDivisionError:
        recall_v = 0
    try: 
        fallout_v = fp / (fp + tn)
    except ZeroDivisionError:
        fallout_v = 0

    try:
        f1 = 2 * (precision_v * recall_v) / (precision_v + recall_v)
    except:
        f1= 0

    return tp, fp, tn, fn, precision_v, recall_v, fallout_v, f1
