from keras.models import Model, Sequential
from keras.layers import Conv1D, BatchNormalization, SpatialDropout1D, LeakyReLU, Input


class TemporalConvNet:
    def __init__(self, seq_length, blocks=9):
        self.seq_length = seq_length
        self.seq_n = 1
        self.blocks = blocks
        self._build(self.blocks)

    def _residual_block(self, factor):
        dilation = 2 ** factor
        inputs = Input(shape=(self.seq_n, self.seq_length,))

        # Residual block
        c1 = Conv1D(self.seq_length, kernel_size=3, strides=1, padding='causal', dilation_rate=dilation)(inputs)
        n1 = BatchNormalization()(c1)
        a1 = LeakyReLU()(n1)
        d1 = SpatialDropout1D(rate=0.2)(a1)
        c2 = Conv1D(self.seq_length, kernel_size=3, strides=1, padding='causal', dilation_rate=dilation)(d1)
        n2 = BatchNormalization()(c2)
        a2 = LeakyReLU()(n2)
        d2 = SpatialDropout1D(rate=0.2)(a2)
        # TODO add skip connections

        return Model(inputs=inputs, outputs=d2)

    def _build(self, blocks):
        model = Sequential()

        for block in range(blocks):
            b = self._residual_block(block)
            model.add(b)

        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        print(model.summary())
