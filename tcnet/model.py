import numpy as np
from keras.models import Model, Sequential
from keras.layers import Conv1D, BatchNormalization, SpatialDropout1D, LeakyReLU, Input, Dense, add, Activation
from keras.optimizers import Adam
from tcnet.activations import gated_activation
from tcnet.metrics import rmse


class TemporalConvNet:
    def __init__(self, seq_length, blocks=9):
        self._seq_length = seq_length
        self._seq_n = 1
        self.optimizer = Adam(lr=0.001)
        self._model = self._build(blocks)
        self._model.compile(optimizer=self.optimizer, loss='mse', metrics=[rmse])

    def _residual_block(self, factor):
        dilation = 2 ** factor
        inputs = Input(shape=(self._seq_length, self._seq_n))

        # Residual block
        c1 = Conv1D(64, kernel_size=3, strides=1, padding='causal', dilation_rate=dilation)(inputs)
        n1 = BatchNormalization(momentum=0.8)(c1)
        a1 = Activation(gated_activation)(n1)
        d1 = SpatialDropout1D(rate=0.2)(a1)
        c2 = Conv1D(64, kernel_size=3, strides=1, padding='causal', dilation_rate=dilation)(d1)
        n2 = BatchNormalization(momentum=0.8)(c2)
        a2 = Activation(gated_activation)(n2)
        d2 = SpatialDropout1D(rate=0.2)(a2)

        # Residual connection
        residual = Conv1D(self._seq_n, kernel_size=1, padding='same')(d2)
        outputs = add([inputs, residual])

        return Model(inputs=inputs, outputs=outputs, name='residual_block_dilation_{}'.format(dilation))

    def _build(self, dilations):
        model = Sequential()

        for dilation in range(dilations):
            block = self._residual_block(dilation)
            model.add(block)

        # model.add(Dense(1, activation='linear'))

        return model

    def print_summary(self):
        # debug only
        print(self._model.summary())

    def train(self, train_x, train_y, val_x, val_y, epochs, verbose):
        return self._model.fit(
            x=train_x,
            y=train_y,
            batch_size=self._seq_length,
            epochs=epochs,
            verbose=verbose
            validation_data=(val_x, val_y)
        )

    def train_batch(self, x, y):
        return self._model.train_on_batch(x, y)

    def predict(self, data, n_ahead):
        predicted = []

        for _ in range(n_ahead):
            pred = self._model.predict(data)
            new = pred[:, -1, :]
            predicted.append(new)
            data = np.append(data, new).reshape(1, -1, 1)
            data = data[:, 1:, :]

        return np.array(predicted).flatten()
