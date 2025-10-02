import os

import numpy
from sklearn.preprocessing import LabelEncoder

from rainwater import sequences_to_series_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class LSTMClassifier:
    """
    Wrapper to make LSTM usable as a sklearn-like function
    You have to override the encode_sequences method to transform the output into something the LSTM can use
    """

    def __init__(self, seq_length: int = 3, epochs: int = 20, batch_size: int = 32):
        """
        Constructor of the LSTM object.
        :param seq_length:
        :param epochs:
        :param batch_size:
        """
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.unique_labels = None
        self.le = LabelEncoder()

    def make_simpler_lstm_model(self, input_shape):
        """
        Puccetti's implementation
        :param input_shape:
        :return:
        """
        input_layer = tf.keras.layers.Input(shape=input_shape)
        lstm = tf.keras.layers.LSTM(64, return_sequences=False, name='lstm')(input_layer)
        dense = tf.keras.layers.Dense(64, activation='relu')(lstm)
        dropout = tf.keras.layers.Dropout(0.3)(dense)
        output = tf.keras.layers.Dense(len(self.unique_labels), activation='softmax')(dropout)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output)
        return model

    def fit(self, X, y):
        """
        Typical sklearn-like fit function
        :param X:
        :param y:
        :return:
        """
        # Transform input to LSTM shape
        X_enc, y_enc = self.encode_sequences(X)
        # Encode Label
        y_enc = self.le.fit_transform(y_enc)
        self.unique_labels = self.le.classes_
        # Build Model
        self.model = self.make_simpler_lstm_model(input_shape=X_enc.shape[1:])
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
        # Train Model
        self.model.fit(X_enc, y_enc, batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks,
                       validation_split=0.2, verbose=1)

    def predict_proba(self, X):
        """
        Outputs (softmax) probabilities
        :param X:
        :return:
        """
        X_enc = self.encode_sequences(X)
        preds = self.model.predict(X_enc, batch_size=self.batch_size)
        return numpy.asarray(preds)

    def predict(self, X):
        """
        Typical sklearn-like predict function
        :param X:
        :return:
        """
        preds = self.predict_proba(X)
        return self.unique_labels[numpy.argmax(preds, axis=1)]

    def encode_sequences(self, X):
        """
        Function to be overridden that transfroms the input into  sequence-like thing the LSTM can process
        :param X:
        :return:
        """
        return X, None


class RainwaterLSTM(LSTMClassifier):
    """
    Overridden class for Rainwater
    """

    def __init__(self, seq_length: int = 3, epochs: int = 20, batch_size: int = 32):
        """
        Constructor of the LSTM object.
        :param seq_length:
        :param epochs:
        :param batch_size:
        """
        super.__init__(seq_length, epochs, batch_size)

    def encode_sequences(self, X):
        """
        Function to be overridden that transfroms the input into  sequence-like thing the LSTM can process
        :param X:
        :return:
        """
        data, label = sequences_to_series_dataset(X, self.seq_length, False)
        return data, label
