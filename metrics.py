import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as conf_m
from sklearn.metrics import precision_score, accuracy_score, recall_score, precision_recall_curve, auc
from sklearn import metrics

import matplotlib.pyplot as plt


def eval_metrics(y_true, y_pred, y_proba, multiclass=True, n_class=3):


    if multiclass:
        d = {}
        for i in range(n_class):
            d[i] = []
            for item in y_true:
                if item == i:
                    d[i].append(1)
                else:
                    d[i].append(0)

        auc_roc = 0
        auc_pr = 0
        for key in d.keys():
            precision_auc, recall_auc, _ = precision_recall_curve(d[key], y_proba[:, key])
            auc_pr = auc_pr + auc(recall_auc, precision_auc)

            fpr, tpr, _ = metrics.roc_curve(d[key], y_proba[:, key])
            auc_roc = auc_roc + metrics.auc(fpr, tpr)

        auc_pr = auc_pr / n_class
        auc_roc = auc_roc / n_class

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

    else:

        fpr, tpr, _ = metrics.roc_curve(y_true, y_proba[:, 1])
        precision_auc, recall_auc, _ = precision_recall_curve(y_true, y_proba[:, 1])

        auc_roc = metrics.roc_auc_score(y_true, y_proba[:, 1])
        # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc_roc))
        # plt.legend(loc=4)
        # plt.show()

        auc_pr = metrics.auc(recall_auc, precision_auc)
        # plt.plot(precision_auc, recall_auc, label="data 1, auc=" + str(auc_pr))
        # plt.legend(loc=4)
        # plt.show()

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

    return acc, precision, recall, auc_pr, auc_roc


    # print('Accuracy: ' + str(np.average(ACC)) +
    #       ' Precision: ' + str(np.average(PPV)) +
    #       ' Specificity: ' + str(np.average(TNR)) +
    #       ' Sensitivity: ' + str(np.average(TPR)))

    # tn, fp, fn, tp = conf_m(y_true, y_pred).ravel()
    # tnr = tn / (tn + fp)



    # print('Accuracy: ' + str(accuracy_score(y_true, y_pred)) +
    #       ' Precision: ' + str(precision_score(y_true, y_pred)) +
    #       ' Specificity: ' + str(tnr) +
    #       ' Sensitivity: ' + str(recall_score(y_true, y_pred)) +
    #       ' AUC ROC: ' + str(auc_roc),
    #       ' PR ROC: ' + str(auc_pr))

    # return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), tnr, \
    #        recall_score(y_true, y_pred), auc_roc, auc_pr





# y_test = np.array([0, 1, 1])
# y_proba = np.array([[0.5, 0.5], [0.4, 0.6], [0.3, 0.7]])
# y_pred = np.array([0, 1, 0])
#
# eval_metrics(y_test, y_pred, y_proba)


# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score
#
# X, y = load_breast_cancer(return_X_y=True)
# clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
# xx_proba = clf.predict_proba(X)
# xx_pred = clf.predict(X)
# yy = clf.predict_proba(X)[:, 1]
#
# eval_metrics(y, xx_pred, xx_proba)
# print(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
# roc_auc_score(y, clf.decision_function(X))
