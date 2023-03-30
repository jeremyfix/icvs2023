import tensorflow as tf
from tensorflow.keras import layers, Sequential


class SegNetencode(tf.keras.layers.Layer):
    def __init__(self):
        super(SegNetencode, self).__init__()
        # Bloc 1
        self.conv1 = Sequential(
            [
                layers.Conv2D(32, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(32, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
            ]
        )
        self.pool1 = layers.MaxPooling2D()

        # Bloc 2
        self.conv2 = Sequential(
            [
                layers.Conv2D(48, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(48, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
            ]
        )
        self.pool2 = layers.MaxPooling2D()

        # Bloc 3
        self.conv3 = Sequential(
            [
                layers.Conv2D(64, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(64, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
            ]
        )
        self.pool3 = layers.MaxPooling2D()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 8, input_shape[2] // 8, 64)

    def build(self, input_shape):
        super(SegNetencode, self).build(input_shape)

    def call(self, inputs):

        x1 = self.conv1(inputs)
        x2 = self.pool1(x1)

        x3 = self.conv2(x2)
        x4 = self.pool2(x3)

        x5 = self.conv3(x4)
        output = self.pool3(x5)

        return output


class SegNetdecode(tf.keras.layers.Layer):
    def __init__(self):
        super(SegNetdecode, self).__init__()

        # Bloc 4
        self.up1 = layers.UpSampling2D()
        self.convtranspose1 = Sequential(
            [
                layers.Conv2D(64, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(64, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
            ]
        )

        # Bloc 3
        self.up2 = layers.UpSampling2D()
        self.convtranspose2 = Sequential(
            [
                layers.Conv2D(48, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(48, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
            ]
        )

        # Bloc 2
        self.up3 = layers.UpSampling2D()
        self.convtranspose3 = Sequential(
            [
                layers.Conv2D(32, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Conv2D(32, 3, padding="same", activation=None),
                layers.BatchNormalization(),
                layers.Activation("relu"),
            ]
        )

        # Output
        # self.output = layers.Conv2DTranspose(1, 1, padding='same', activation='sigmoid')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 8, input_shape[2] * 8, 32)

    def build(self, input_shape):
        # self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim),initializer='uniform',trainable=True)
        super(SegNetdecode, self).build(input_shape)

    def call(self, inputs):

        x1 = self.up1(inputs)
        x2 = self.convtranspose1(x1)

        x3 = self.up2(x2)
        x4 = self.convtranspose2(x3)

        x5 = self.up3(x4)
        x6 = self.convtranspose3(x5)

        # x7 = self.up4(x6)
        # x8 = self.convtranspose4(x7)

        # output = self.output(x8)

        # output = self.project(x9)

        return x6


def SegNetSeq(seq_size, img_height, img_width):
    model = Sequential(
        [
            layers.TimeDistributed(
                SegNetencode(), input_shape=(seq_size, img_height, img_width, 1)
            ),
            layers.TimeDistributed(SegNetdecode()),
            layers.TimeDistributed(
                layers.Conv2DTranspose(2, 1, padding="same", activation="sigmoid")
            ),
        ]
    )
    return model


if __name__ == "__main__":
    model = SegNetSeq(40, 192, 192)
    model.summary()
