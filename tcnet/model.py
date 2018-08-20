from keras.models import Model
from keras.layers import Conv1D, BatchNormalization, Dropout, SpatialDropout1D, LeakyReLU, Input, Dense


class TemporalConvNet:
    def __init__(self, seq_length, blocks=9):
        self._seq_length = seq_length
        self._seq_n = 1
        self._blocks = blocks
        self._model = self._build(self._blocks)
        self._model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    def _residual_block(self, input_tensor, factor):
        dilation = 2 ** factor

        # Residual block
        c1 = Conv1D(self._seq_length, kernel_size=3, strides=1, padding='causal', dilation_rate=dilation)(input_tensor)
        n1 = BatchNormalization(momentum=0.8)(c1)
        a1 = LeakyReLU(alpha=0.2)(n1)
        d1 = SpatialDropout1D(rate=0.2)(a1)
        c2 = Conv1D(self._seq_length, kernel_size=3, strides=1, padding='causal', dilation_rate=dilation)(d1)
        n2 = BatchNormalization(momentum=0.8)(c2)
        a2 = LeakyReLU(alpha=0.2)(n2)
        d2 = SpatialDropout1D(rate=0.2)(a2)
        # TODO add skip connections

        return d2

    def _build(self, blocks):
        input_layer = Input(shape=(self._seq_length, self._seq_n))
        initial = input_layer

        for block in range(blocks):
            initial = self._residual_block(initial, block)

        dense = Dense(self._seq_length, activation='relu')(initial)
        drop = Dropout(rate=0.2)(dense)
        output = Dense(1, activation='linear')(drop)

        return Model(inputs=input_layer, outputs=output)

    def print_summary(self):
        # debug only
        print(self._model.summary())

    def train(self, train_x, train_y, val_x, val_y, epochs, verbose):
        return self._model.fit(
            x=train_x,
            y=train_y,
            batch_size=self._seq_length,
            epochs=epochs,
            verbose=verbose,
            validation_data=(val_x, val_y)
        )

    def train_batch(self, x, y):
        return self._model.train_on_batch(x, y)

    def predict(self):
        pass
