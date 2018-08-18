from keras.models import Model
from keras.layers import Conv1D, BatchNormalization, SpatialDropout1D, LeakyReLU, Input, Dense


class TemporalConvNet:
    def __init__(self, seq_length, blocks=9):
        self.seq_length = seq_length
        self.seq_n = 1
        self.blocks = blocks
        self.model = self._build(self.blocks)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    def _residual_block(self, input_tensor, factor):
        dilation = 2 ** factor

        # Residual block
        c1 = Conv1D(self.seq_length, kernel_size=3, strides=1, padding='causal', dilation_rate=dilation)(input_tensor)
        n1 = BatchNormalization(momentum=0.8)(c1)
        a1 = LeakyReLU(alpha=0.2)(n1)
        d1 = SpatialDropout1D(rate=0.2)(a1)
        c2 = Conv1D(self.seq_length, kernel_size=3, strides=1, padding='causal', dilation_rate=dilation)(d1)
        n2 = BatchNormalization(momentum=0.8)(c2)
        a2 = LeakyReLU(alpha=0.2)(n2)
        d2 = SpatialDropout1D(rate=0.2)(a2)
        # TODO add skip connections

        return d2

    def _build(self, blocks):
        input_layer = Input(shape=(self.seq_length, self.seq_n))
        initial = input_layer

        for block in range(blocks):
            initial = self._residual_block(initial, block)

        # TODO add one more dense layer
        output = Dense(1, activation='linear')(initial)
        return Model(inputs=input_layer, outputs=output)

    def print_summary(self):
        # debug only
        print(self.model.summary())

    def train(self):
        pass

    def predict(self):
        pass
