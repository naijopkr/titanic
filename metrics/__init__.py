from sklearn.metrics import confusion_matrix, classification_report
from pandas import Series

def print_cm(y_true: Series, y_pred: Series):
    pred_false, pred_true = confusion_matrix(y_true, y_pred)
    tn, fn = pred_false
    fp, tp = pred_true
    print(
        f'TN\tFN\tFP\tTP\n{tn}\t{fn}\t{fp}\t{tp}'
    )

def print_cr(y_true: Series, y_pred: Series):
    print(classification_report(y_true, y_pred))
