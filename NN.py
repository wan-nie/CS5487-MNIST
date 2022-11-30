import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from utils import plot_history
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random


def get_model(kernel_size, dropout_rate):
    model = keras.Sequential([
        Conv2D(16, kernel_size, strides=(2,2), activation='relu', input_shape=(28, 28, 1), padding='same'),
        Conv2D(32, kernel_size, strides=(2,2), activation='relu', padding='same'),
        Conv2D(64, kernel_size, strides=(1,1), activation='relu', padding='same'),
        Dropout(rate=dropout_rate),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(rate=dropout_rate),
        Dense(10, activation='softmax')
    ])

    return model


def train_model(my_model, train_x, train_y, datagen=False, noise=False, verbose=0, save_path=None):
    # shuffle and split data
    train_x, train_y = shuffle(train_x, train_y)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)

    # compile model
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    my_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    # callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=args.ReduceLR_Patience, verbose=verbose,
                          factor=args.ReduceLR_Factor),
        EarlyStopping(monitor='val_loss', patience=args.EarlyStop_Patience),
    ]
    if save_path:
        callbacks.append(ModelCheckpoint(filepath=save_path, monitor='val_acc',
                                         save_best_only=True))

    if not datagen:
        history = my_model.fit(
            x=train_x,
            y=train_y,
            validation_data=(val_x, val_y),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
    else:
        def add_gauss_noise(data, sigma=0.2):
            return data + np.random.normal(0, sigma, data.shape)

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            preprocessing_function=add_gauss_noise if noise else None,
        )

        datagen.fit(train_x)
        history = my_model.fit(
            x=datagen.flow(train_x, train_y, batch_size=args.batch_size),
            validation_data=(val_x, val_y),
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=verbose
        )

    return history


def hyperparams_search(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)  # for reproducibility
    result_ls = []
    kernel_size_ls = [(3,3), (4,4), (5,5)]
    dropout_rate_ls = [0.3, 0.5]
    for k in kernel_size_ls:
        for r in dropout_rate_ls:
            keras.backend.clear_session()
            model1 = get_model(kernel_size=k, dropout_rate=r)
            model2 = get_model(kernel_size=k, dropout_rate=r)
            h1 = train_model(model1, data1.train_x, data1.train_y, verbose=0)
            h2 = train_model(model2, data2.train_x, data2.train_y, verbose=0)
            score1 = np.max(h1.history['val_acc'])
            score2 = np.max(h2.history['val_acc'])
            result_ls.append([k, r, score1, score2])
    df = pd.DataFrame(result_ls, columns=['kernel_size', 'dropout_rate', 'score1', 'score2'])
    df.to_csv('./result/NN_hyperparams_search_result.csv', index=False)


def eval_on_test_and_challenge(datagen=False, noise=False, seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)  # for reproducibility
    result_ls = []
    for i, data in enumerate([data1, data2]):
        keras.backend.clear_session()
        save_path = f'./result/models/model{i}_datagen{datagen}_noise{noise}.hdf5'
        model = get_model(kernel_size=(4,4), dropout_rate=0.5)
        print(model.summary())
        train_model(model, train_x=data.train_x, train_y=data.train_y, datagen=datagen,
                    verbose=1, noise=noise, save_path=save_path)
        best_model = keras.models.load_model(save_path)
        score = best_model.evaluate(data.test_x, data.test_y)[1]
        c_score = best_model.evaluate(c_X, c_y)[1]
        result_ls.append([i, score, c_score])
    df = pd.DataFrame(result_ls, columns=['trial', 'score', 'challenge score'])
    df.to_csv(f'./result/NN_2trials_result_datagen{datagen}_noise{noise}.csv', index=False)


if __name__ == '__main__':
    X = np.load('./data/processed_data/vectors.npy')
    y = np.load('./data/processed_data/labels.npy')
    c_X = np.load('./data/processed_data/cvectors.npy')
    c_y = np.load('./data/processed_data/clabels.npy')
    #
    X = np.expand_dims(X, axis=-1) / 255.
    c_X = np.expand_dims(c_X, axis=-1) / 255.
    y = keras.utils.to_categorical(y, num_classes=10)
    c_y = keras.utils.to_categorical(c_y, num_classes=10)

    data1 = {
        'train_x': X[:2000],
        'test_x': X[2000:],
        'train_y': y[:2000],
        'test_y': y[2000:]
    }
    data1 = SimpleNamespace(**data1)

    data2 = {
        'train_x': X[2000:],
        'test_x': X[:2000],
        'train_y': y[2000:],
        'test_y': y[:2000]
    }
    data2 = SimpleNamespace(**data2)

    args = {
        'epochs': 100,
        'batch_size': 32,
        'ReduceLR_Patience': 5,
        'ReduceLR_Factor': 0.9,
        'EarlyStop_Patience': 10,
        'lr': 1e-3,
    }
    args = SimpleNamespace(**args)

    # search for best combination of kernel_size (4,4) and dropout rate (0.5)
    # record in './result/'
    # hyperparams_search()

    # eval model on test set and challenge test set
    eval_on_test_and_challenge(datagen=False, noise=False)
    eval_on_test_and_challenge(datagen=True, noise=False)
    eval_on_test_and_challenge(datagen=True, noise=True)






