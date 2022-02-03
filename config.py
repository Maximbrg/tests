from tab2img.converter import Tab2Img

import pandas as pd
import numpy as np
NUM_FOLDS_OUTTER = 10

def data_loader():
    d0 = 'datasets/lymphography.csv'
    d1 = 'datasets/synthetic-control.csv'
    d2 = 'datasets/glass.csv'
    d3 = 'datasets/heart-va.csv'
    d4 = 'datasets/yeast.csv'
    d5 = 'datasets/cylinder-bands.csv'
    d6 = 'datasets/molec-biol-promoter.csv'
    d7 = 'datasets/spambase.csv'
    d8 = 'datasets/wine-quality-red.csv'
    d9 = 'datasets/wine-quality-white.csv'
    d10 = 'datasets/oocytes_merluccius_states_2f.csv'
    d11 = 'datasets/teaching.csv'
    d12 = 'datasets/credit-approval.csv'
    d13 = 'datasets/energy-y2.csv'
    d14 = 'datasets/hepatitis.csv'
    # data_names = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14]#, d1, d2, d3]
    data_names = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14]#, d1, d2, d3]
    data_frames = []
    for csv_name in data_names:
        temp_df = pd.read_csv(csv_name)
        temp_df = temp_df.rename(columns={'clase': 'class', 'symboling': 'class', 'Hall_of_Fame': 'class'},
                                 inplace=False)
        temp_df = temp_df.fillna(temp_df.mean())
        for col_name in temp_df.columns:
            if temp_df[col_name].dtype == "object":
                temp_df[col_name] = pd.Categorical(temp_df[col_name])
                temp_df[col_name] = temp_df[col_name].cat.codes
        X = temp_df.drop('class', axis=1)
        y = temp_df['class']
        data_frames.append((X, y, len(pd.unique(temp_df['class']))))

    return data_frames


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def one_hot(y_test, n_class):
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1, 1)
    y_test = indices_to_one_hot(y_test, n_class)

    return y_test


class Data:
    pass

# dataframes = data_loader()
# k = 0
# for df in dataframes:
#     image_convertor = Tab2Img()
#
#     features = df[0].values
#     target = df[1].values
#     n_class = df[2]
#
#
#     x_train_images = image_convertor.fit_transform(features, target)
#     x_train_images = (np.repeat(x_train_images[..., np.newaxis], 3, -1))
#     print(k, x_train_images.shape)
#     k = k + 1

