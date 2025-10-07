import os.path

import numpy
import sklearn

from rainwater import get_injectors, sequences_to_series_dataset
from rainwater.ModelTrainer import ModelTrainer
from rainwater.utils import read_csv, load_dataset_config, print_full_sequences, read_general_conf

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def make_simpler_lstm_model(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    lstm = tf.keras.layers.LSTM(64, return_sequences=False, name='lstm')(input_layer)
    dense = tf.keras.layers.Dense(64, activation='relu')(lstm)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    output = tf.keras.layers.Dense(2, activation='softmax')(dropout)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    return model


class LSTMClassifier:

    def __init__(self, seq_length: int = 3, epochs: int = 20, batch_size: int = 32):
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.unique_labels = None

    def fit(self, X, y):
        self.unique_labels = numpy.unique(y)
        self.model = make_simpler_lstm_model(input_shape=X.shape[1:])
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=0.0001),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=1),
        ]
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks,
                                 validation_split=0.2, verbose=1)

    def predict_proba(self, X):
        preds = self.model.predict(X, batch_size=self.batch_size)
        return numpy.asarray(preds)

    def predict(self, X):
        preds = self.predict_proba(X)
        return numpy.argmax(preds, axis=1)


if __name__ == '__main__':

    # Reading general info
    general_cfg = read_general_conf('general_cfg.cfg')

    # Set the model trainer once
    model_trainer = ModelTrainer(policy=general_cfg['policy'],
                                 tt_split=general_cfg['tt_split'],
                                 force_binary=general_cfg['force_binary'])

    # Iterating over input files / case studies
    for cfg_file in os.listdir(general_cfg['input_folder']):

        if cfg_file.endswith('.cfg'):

            dataset_info = load_dataset_config(os.path.join(general_cfg['input_folder'], cfg_file))
            sequences = read_csv(dataset_info['csv_path'], dataset_info['kwh_col'],
                                 dataset_info['time_col'], dataset_info['seq_col'],
                                 dataset_info['batch_size'], dataset_info['limit_rows'])
            print(dataset_info['csv_path'])

            # Injecting anomalies
            injected_sequences = []
            print('\nPerforming Injection...')
            for injector in get_injectors(dataset_info['threats'], dataset_info['duration'],
                                          dataset_info['min_normal'], dataset_info['perc_inj']):
                for seq in sequences:
                    new_s = injector.inject(seq)
                    if new_s is not None and len(numpy.unique(new_s.labels)) >= 2:
                        injected_sequences.append(new_s)

            data, label = sequences_to_series_dataset(injected_sequences, 3, False)
            label = numpy.where(label == 'normal', 0, 1)
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, label, test_size=0.4, shuffle=False)
            print("OK")
            model = LSTMClassifier()
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            print("BOH")
            print(sklearn.metrics.balanced_accuracy_score(preds, y_test))
