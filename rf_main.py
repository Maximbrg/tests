import config
import models.simple_fcnn as fcnn_lib
from sklearn.model_selection import KFold
import metrics
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split


from tab2img.converter import Tab2Img

dataframes = config.data_loader()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

kfold_outter = KFold(n_splits=config.NUM_FOLDS_OUTTER, shuffle=True)

d = {'Accuracy': [], 'Precision': [], 'Recall': [],
           'AUC ROC': [], 'auc pr': []}

df_marks = pd.DataFrame(d)

for df in dataframes:
    print(1)
    features = df[0].values
    target = df[1].values
    n_class = df[2]

    # x_train, x_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    for train, test in kfold_outter.split(features, target):
        # print('1')
        x_train = features[train]
        Y_train = target[train]
        x_test = features[test]
        Y_test = target[test]

        clf = RandomForestClassifier()
        clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, cv=3, verbose=0,
                                 random_state=42)
        clf = clf.fit(x_train, Y_train)

        Y_proba = clf.predict_proba(x_test)
        Y_pred = clf.predict(x_test)
        if n_class > 2:
            multiclass = True
        else:
            multiclass = False
        acc, precision, recall, auc_pr, auc_roc = \
            metrics.eval_metrics(Y_test, Y_pred, Y_proba, multiclass=multiclass, n_class=n_class)

        new_row = {'Accuracy': acc, 'Precision': precision, 'Recall': recall,
                   'AUC ROC': auc_roc, 'auc pr': auc_pr}
        df_marks = df_marks.append(new_row, ignore_index=True)

    # for train, test in kfold_outter.split(features, target):
    #     print('1')
    #     x_train = features[train]
    #     Y_train = target[train]
    #     x_test = features[test]
    #     Y_test = target[test]
    #
    #     clf = RandomForestClassifier()
    #     clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, cv=3, verbose=0,
    #                              random_state=42)  # 3- folds
    #
    #     clf = clf.fit(x_train, Y_train)
    #
    #     Y_proba = clf.predict_proba(x_test)
    #     Y_pred = clf.predict(x_test)
    #
    #     acc, precision, specificity, sensitivity, aucroc, aucpr = \
    #         metrics.eval_metrics(Y_test, Y_pred, Y_proba)
    #
    #     new_row = {'Accuracy': acc, 'Precision': precision, 'Specificity': specificity, 'Sensitivity': sensitivity,
    #                'AUC ROC': aucroc, 'auc pr': aucpr}
    #     df_marks = df_marks.append(new_row, ignore_index=True)


df_marks.to_csv('results\\mammographic.csv')