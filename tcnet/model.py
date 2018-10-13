import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Conv1D, BatchNormalization, Input, Activation, add
from keras.optimizers import Adam
from keras.utils import CustomObjectScope
from tcnet.activations import gated_activation
from tcnet.metrics import rmse


class TemporalConvNet:
    def __init__(self, seq_length, blocks=6, filters=64):
        self._seq_length = seq_length
        self._n_filters = filters
        self._optimizer = Adam(lr=0.001)
        self._model = self._build(blocks)
        self._model.compile(optimizer=self._optimizer, loss='mse', metrics=[rmse])

    def _residual_block(self, factor):
        dilation = 2 ** factor
        inputs = Input(shape=(self._seq_length, 1))

        # Residual block
        c1 = Conv1D(self._n_filters, kernel_size=4, strides=1, padding='causal', dilation_rate=dilation)(inputs)
        a1 = Activation(gated_activation)(c1)
        n1 = BatchNormalization(momentum=0.6)(a1)
        c2 = Conv1D(self._n_filters, kernel_size=4, strides=1, padding='causal', dilation_rate=dilation)(n1)
        a2 = Activation(gated_activation)(c2)
        n2 = BatchNormalization(momentum=0.6)(a2)

        # Residual connection
        residual = Conv1D(1, kernel_size=1, padding='same')(n2)
        outputs = add([inputs, residual])

        return Model(inputs=inputs, outputs=outputs, name='residual_block_{}'.format(factor))

    def _build(self, dilations):
        model = Sequential()

        for dilation in range(dilations):
            block = self._residual_block(dilation)
            model.add(block)

        return model

    def train(self, train_x, train_y, epochs, verbose, x_test):
        return self._model.fit(
            x=train_x,
            y=train_y,
            batch_size=self._seq_length,
            epochs=epochs,
            verbose=verbose
        )

    def predict(self, data, n_ahead):
        predicted = []

        for _ in range(n_ahead):
            pred = self._model.predict(data)
            new = pred[:, -1, :]
            predicted.append(new)
            data = np.append(data, new).reshape(1, -1, 1)
            data = data[:, 1:, :]

        return np.array(predicted).flatten()

    def save_model(self, name):
        self._model.save('{}.h5'.format(name))
        print('Model has been saved as {}.h5'.format(name))

    def load_model(self, path):
        with CustomObjectScope({'gated_activation': gated_activation, 'rmse': rmse}):
            self._model = load_model(path)
            print('Model restored')
