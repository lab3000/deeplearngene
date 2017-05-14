import pandas as pd
import numpy as np


def onehot_misclassified(model, x, y):
    '''input: keras model, x data, y data
      output: a dictionary of misclassified
      datapoints

      y should be a categorical representation
      as a numpy array with row-wise onehot encoded
      vectors'''
    y_pred = model.predict_classes(x)
    y_true = np.apply_along_axis(np.argmax, 1, y)

    mis_classified = {}
    mis_classified['true_class'] = []
    mis_classified['pred_class'] = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            mis_classified['true_class'].append(y_true[i])
            mis_classified['pred_class'].append(y_pred[i])

    return mis_classified
