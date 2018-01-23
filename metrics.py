import numpy as np

# shape = (instances, labels)

def count_by_labels(nparray):
    return np.sum(nparray, axis = 0)

def true_positive(y_true, y_pred):
    return count_by_labels(y_true * y_pred)

def false_positive(y_true, y_pred):
    return count_by_labels(np.clip(y_pred - y_true, 0, 1))

def true_negative(y_true, y_pred):
    return true_positive(1 - y_true, 1 - y_pred)

def false_negative(y_true, y_pred):
    return false_positive(1 - y_true, 1 - y_pred)

def accuracy(tp, fp, tn, fn): return (tp + tn) / (tp + fp + tn + fn)
def precision(tp, fp, tn, fn): return tp / (tp + fp) if tp + fp > 0 else 1
def recall(tp, fp, tn, fn): return tp / (tp + fn) if tp + fn > 0 else 1

def micro(metric):
    def f(y_true, y_pred):
        tp = np.mean(true_positive(y_true, y_pred))
        fp = np.mean(false_positive(y_true, y_pred))
        tn = np.mean(true_negative(y_true, y_pred))
        fn = np.mean(false_negative(y_true, y_pred))

        return metric(tp, fp, tn, fn)

    return f

def macro(metric):
    def f(y_true, y_pred):
        tp = true_positive(y_true, y_pred)
        fp = false_positive(y_true, y_pred)
        tn = true_negative(y_true, y_pred)
        fn = false_negative(y_true, y_pred)

        matrix = np.vstack((tp, fp, tn, fn))
        unpacked = lambda a: metric(*a)
        return np.mean(np.apply_along_axis(unpacked, 0, matrix))

    return f

def fmeasure(precision, recall):
    return 2 * precision * recall / (precision + recall)

# def accuracy(y_true, y_pred):
#     intsec = _intersection(y_true, y_pred)
#     union = np.sum(np.maximum(y_true, y_pred), axis = 0)

#     # fractions with 0 in denominator become 1
#     # as in mlR and Mulan implementations
#     div = np.where(union == 0, 1, intsec / union)
#     return np.mean(div)


# def _intersection_mean(divisor_f):
#     def metric(y_true, y_pred, ignore_nan = True, nan_value = 1):
#         divisor = divisor_f(y_true, y_pred)
        
#         if ignore_nan:
#             return np.nanmean(_intersection(y_true, y_pred) / divisor)
#         else:
#             fractions = np.where(np.isnan(_intersection(y_true, y_pred) / divisor),
#                                  nan_value,
#                                  divisor)
#             return np.mean(fractions)
#     return metric

# precision = _intersection_mean(lambda y_true, y_pred: np.sum(y_true, axis = 0))
# recall = _intersection_mean(lambda y_true, y_pred: np.sum(y_pred, axis = 0))


# def fmeasure(y_true, y_pred, ignore_nan = True):
#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     return 2 * p * r / (p + r)

def hamming_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def subset_accuracy(y_true, y_pred):
    return np.mean(np.all(y_true == y_pred, axis = 1))

def report(y_true, y_pred):
    metrics = {
        "microP": micro(precision),
        "microR": micro(recall),
        "microA": micro(accuracy),
        "microF": lambda t, p: fmeasure(metrics["microP"](t, p), metrics["microR"](t, p)),
        "macroP": macro(precision),
        "macroR": macro(recall),
        "macroA": macro(accuracy),
        "macroF": lambda t, p: fmeasure(metrics["macroP"](t, p), metrics["macroR"](t, p)),
        "hamming": hamming_loss,
        "subsetA": subset_accuracy
    }

    for name, f in metrics.items():
        #print("Calculating {}".format(name))
        print("{}: {}".format(name, f(y_true, y_pred)))
    
