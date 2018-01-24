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
def precision(tp, fp, tn, fn): return tp / (tp + fp)# if tp + fp > 0 else 1
def recall(tp, fp, tn, fn): return tp / (tp + fn)# if tp + fn > 0 else 1

def micro(metric):
    def f(y_true, y_pred):
        tp = np.sum(true_positive(y_true, y_pred))
        fp = np.sum(false_positive(y_true, y_pred))
        tn = np.sum(true_negative(y_true, y_pred))
        fn = np.sum(false_negative(y_true, y_pred))
        # print(np.array([[tp, fp],[fn, tn]]))
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
        applied = np.apply_along_axis(unpacked, 0, matrix)
        # print(applied)
        # ignore nans for mean:
        return np.nanmean(applied)

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

confusion_matrix = micro(lambda tp, fp, tn, fn: np.array([[tp, fp], [fn, tn]]))

def all_metrics():
    metrics = {
        "accuracy": micro(accuracy),
        "microP": micro(precision),
        "microR": micro(recall),
        "microF": lambda t, p: fmeasure(metrics["microP"](t, p), metrics["microR"](t, p)),
        "macroP": macro(precision),
        "macroR": macro(recall),
        "macroF": lambda t, p: fmeasure(metrics["macroP"](t, p), metrics["macroR"](t, p)),
        "hamming": hamming_loss,
        "subsetA": subset_accuracy
    }
    return metrics

def report(y_true, y_pred):
    print("Confusion matrix:\n{}".format(confusion_matrix(y_true, y_pred)))
    
    for name, f in all_metrics().items():
        #print("Calculating {}".format(name))
        print("{}: {}".format(name, f(y_true, y_pred)))
    
def csv_report(filename, title, y_true, y_pred, val_step = None):
    metrics = all_metrics()
    
    try:
        f = open(filename)
        f.close()
    except IOError as e:
        with open(filename, "w") as outfile:
            for name in metrics.keys():
                outfile.write("{}, ".format(name))

            if val_step:
                outfile.write("val_step,")
                
            outfile.write("title\n")
            
    with open(filename, "a") as outfile:
        for f in metrics.values():
            outfile.write("{}, ".format(f(y_true, y_pred)))

        if val_step:
            outfile.write("{},".format(val_step))
                
        outfile.write("{}\n".format(title))
