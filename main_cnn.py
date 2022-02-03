import config
import models.simple_fcnn as fcnn_lib
import models.cnn as cnn_lib
from models import Distiller
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import copy

import metrics
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential, Model

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
k = 0
for df in dataframes:
    features = df[0].values
    target = df[1].values
    n_class = df[2]

    if n_class > 2:
        multiclass = True
    else:
        multiclass = False

    if k == 0 or k == 10 or k == 14:
        k = k + 1
        UP = 7
        SIZE = 35
    elif k == 1 or k == 6 or k == 7:
        k = k + 1
        UP = 4
        SIZE = 32
    elif k == 2 or k == 4 or k == 11 or k == 13:
        k = k + 1
        UP = 12
        SIZE = 36
    elif k == 3 or k == 8 or k == 9 or k == 12:
        k = k + 1
        UP = 8
        SIZE = 32
    elif k == 5:
        k = k + 1
        UP = 6
        SIZE = 36


    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    for train, test in kfold_outter.split(features, target):
        print(0)
        image_convertor = Tab2Img()

        X_train = features[train]
        y_train = target[train]
        X_test = features[test]
        y_test = target[test]

        x_train_images = image_convertor.fit_transform(X_train, y_train)
        x_test_images = image_convertor.transform(X_test)
        x_train_images = (np.repeat(x_train_images[..., np.newaxis], 3, -1))
        x_test_images = (np.repeat(x_test_images[..., np.newaxis], 3, -1))

        Y_train_onehot = config.one_hot(y_train, n_class)
        Y_test_onehot = config.one_hot(y_test, n_class)

        teacher = cnn_lib.convolutional_neural_network(UP, UP, SIZE, SIZE, num_classes=n_class)

        clallback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )

        teacher.fit(x_train_images, Y_train_onehot, batch_size=16, validation_split=0.1, epochs=20, verbose=1, callbacks=[clallback])
        best_acurracy = 0
        models = []
        for i in range(5):

            student = fcnn_lib.fully_fcnn(n_class=n_class)
            models.append(Distiller.Distiller(student=student, teacher=teacher))
            distiller = models[i]
            distiller.compile(
                optimizer='adam',
                metrics=['accuracy'],
                student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=False),
                distillation_loss_fn=keras.losses.KLDivergence(),
                alpha=0.1,
                temperature=15,
            )

            distiller.fit([x_train_images, X_train], Y_train_onehot, epochs=50, batch_size=16)

            teacher = distiller.student
            x_train_images = X_train

            Y_proba = distiller.predict(X_test)
            Y_pred = Y_proba.argmax(axis=1)

            acc, precision, recall, auc_pr, auc_roc = \
                    metrics.eval_metrics(y_test, Y_pred, Y_proba, multiclass=multiclass, n_class=n_class)

            if acc > best_acurracy:
                best_model = models[i]


        #########################################################
        Y_proba = best_model.predict(X_test)
        Y_pred = Y_proba.argmax(axis=1)


        acc, precision, recall, auc_pr, auc_roc = \
                metrics.eval_metrics(y_test, Y_pred, Y_proba, multiclass=multiclass, n_class=n_class)

        new_row = {'Accuracy': acc, 'Precision': precision, 'Recall': recall,
                   'AUC ROC': auc_roc, 'auc pr': auc_pr}
        df_marks = df_marks.append(new_row, ignore_index=True)

        ##########################################################
        # extract = Model(inputs=best_model.student.inputs, outputs=best_model.student.layers[-2].output)
        #
        # features_train = extract.predict(X_train)
        # features_test = extract.predict(X_test)
        #
        # X_train_concatenate = np.concatenate([X_train, features_train], axis=-1)
        # X_test_concatenate = np.concatenate([X_test, features_test], axis=-1)
        #
        # clf = RandomForestClassifier()
        # clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, cv=3, verbose=0,
        #                          random_state=42)
        # clf = clf.fit(X_train_concatenate, y_train)
        #
        # Y_proba = clf.predict_proba(X_test_concatenate)
        # Y_pred = clf.predict(X_test_concatenate)
        #
        # acc, precision, recall, auc_pr, auc_roc = \
        #     metrics.eval_metrics(y_test, Y_pred, Y_proba, multiclass=True, n_class=n_class)
        #
        # new_row = {'Accuracy': acc, 'Precision': precision, 'Recall': recall,
        #            'AUC ROC': auc_roc, 'auc pr': auc_pr}
        # df_marks = df_marks.append(new_row, ignore_index=True)
        ##########################################################

df_marks.to_csv('results\\cnn_results.csv')



#     for train, test in kfold_outter.split(features, target):
#         image_convertor = Tab2Img()
#         x_train = features[train]
#         Y_train = target[train]
#         Y_train_onehot = config.one_hot(target[train], n_class)
#         x_test = features[test]
#         Y_test = target[test]
#         Y_test_onehot = config.one_hot(target[test], n_class)
#
#         x_train_images = image_convertor.fit_transform(x_train, Y_train)
#         x_test_images = image_convertor.transform(x_test)
#
#         x_train_images = (np.repeat(x_train_images[..., np.newaxis], 3, -1))
#         x_test_images = (np.repeat(x_test_images[..., np.newaxis], 3, -1))
#
#         cnn = cnn_lib.convolutional_neural_network(12, 12, 36, 36)
#
#         clallback = tf.keras.callbacks.EarlyStopping(
#             monitor="val_loss",
#             min_delta=0,
#             patience=5,
#             verbose=0,
#             mode="auto",
#             baseline=None,
#             restore_best_weights=True,
#         )
#
#         cnn.fit(x_train_images, Y_train_onehot, batch_size=32, validation_split=0.1, epochs=20, verbose=1, callbacks=[clallback])
#
#         Y_proba = cnn.predict(x_test_images)
#         Y_pred = Y_proba.argmax(axis=1)
#
#         acc, precision, specificity, sensitivity, aucroc, aucpr = \
#             metrics.eval_metrics(Y_test, Y_pred, Y_proba)
#
#         new_row = {'Accuracy': acc, 'Precision': precision, 'Specificity': specificity, 'Sensitivity': sensitivity,
#                    'AUC ROC': aucroc, 'auc pr': aucpr}
#         df_marks = df_marks.append(new_row, ignore_index=True)
#
#
# df_marks.to_csv('results\\cnn_results_hybrid_d0.csv')
