import config
import models.simple_fcnn as fcnn_lib
from sklearn.model_selection import KFold
import metrics
import pandas as pd

from tab2img.converter import Tab2Img

dataframes = config.data_loader()

kfold_outter = KFold(n_splits=config.NUM_FOLDS_OUTTER, shuffle=True)

d = {'Accuracy': [], 'Precision': [], 'Specificity': [], 'Sensitivity': [],
                   'AUC ROC': [], 'auc pr': []}

df_marks = pd.DataFrame(d)

for df in dataframes:

    features = df[0].values
    target = df[1].values
    n_class = df[2]

    for train, test in kfold_outter.split(features, target):
        image_convertor = Tab2Img()
        x_train = features[train]
        Y_train = config.one_hot(target[train], n_class)
        x_test = features[test]
        Y_test = target[test]
        Y_test_onehot = config.one_hot(target[test], n_class)

        # x_train_images = image_convertor.fit_transform(x_train, Y_train)
        # x_test_images = image_convertor.transform(x_test)

        fcnn_model = fcnn_lib.fully_fcnn(n_class=n_class)

        fcnn_model.fit(x_train, Y_train, batch_size=16, validation_split=0.1, epochs=1, verbose=1)

        Y_proba = fcnn_model.predict(x_test)
        Y_pred = Y_proba.argmax(axis=1)

        acc, precision, recall, auc_pr, auc_roc = \
            metrics.eval_metrics(Y_test, Y_pred, Y_proba)

        new_row = {'Accuracy': acc, 'Precision': precision, 'Recall': recall,
                   'AUC ROC': auc_roc, 'auc pr': auc_pr}
        df_marks = df_marks.append(new_row, ignore_index=True)


df_marks.to_csv('results\\fcnn_results.csv')
