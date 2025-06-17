# Scripts to train and perform inference of ConvLSTM/UNet/SegNet
# for predicting knots from the contours of trees
# Copyright (C) 2023 Anonymous

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import tensorflow as tf
from tensorflow.keras import layers, Sequential


class ConvLSTMencode(tf.keras.layers.Layer):
    def __init__(self, droprate):
        super(ConvLSTMencode, self).__init__()

        self.bloc1conv1 = layers.ConvLSTM2D(
            32,
            3,
            padding="same",
            activation="relu",
            return_sequences=True,
            go_backwards=True,
        )
        self.bloc1conv2 = layers.ConvLSTM2D(
            32,
            3,
            padding="same",
            activation="relu",
            return_sequences=True,
            go_backwards=True,
        )
        self.bloc1maxpool = layers.TimeDistributed(layers.MaxPooling2D())

        self.bloc2conv1 = layers.ConvLSTM2D(
            48,
            3,
            padding="same",
            activation="relu",
            return_sequences=True,
            go_backwards=True,
        )
        self.bloc2conv2 = layers.ConvLSTM2D(
            48,
            3,
            padding="same",
            activation="relu",
            return_sequences=True,
            go_backwards=True,
        )
        self.bloc2maxpool = layers.TimeDistributed(layers.MaxPooling2D())

        self.bloc3conv1 = layers.ConvLSTM2D(
            64,
            3,
            padding="same",
            activation="relu",
            return_sequences=True,
            go_backwards=True,
        )
        self.bloc3conv2 = layers.ConvLSTM2D(
            64,
            3,
            padding="same",
            activation="relu",
            return_sequences=True,
            go_backwards=True,
        )
        self.bloc3maxpool = layers.TimeDistributed(layers.MaxPooling2D())

        self.droplayer = layers.TimeDistributed(layers.Dropout(droprate))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 8, input_shape[2] // 8, 64)

    def build(self, input_shape):
        super(ConvLSTMEnc, self).build(input_shape)

    def call(self, inputs):

        x1 = self.bloc1conv1(inputs)
        x2 = self.bloc1conv2(x1)
        x3 = self.bloc1maxpool(x2)

        x4 = self.bloc2conv1(x3)
        x5 = self.bloc2conv2(x4)
        x6 = self.bloc2maxpool(x5)

        x7 = self.bloc3conv1(x6)
        x8 = self.bloc3conv2(x7)
        x9 = self.bloc3maxpool(x8)

        output = self.droplayer(x9, training=True)

        return output


class ConvLSTMdecode(tf.keras.layers.Layer):
    def __init__(self, droprate):
        super(ConvLSTMdecode, self).__init__()

        self.droplayer = layers.TimeDistributed(layers.Dropout(droprate))

        self.bloc1upsamp = layers.TimeDistributed(layers.UpSampling2D())
        self.bloc1conv1 = layers.TimeDistributed(
            layers.Conv2DTranspose(64, 3, padding="same", activation="relu")
        )
        self.bloc1conv2 = layers.TimeDistributed(
            layers.Conv2DTranspose(64, 3, padding="same", activation="relu")
        )

        self.bloc2upsamp = layers.TimeDistributed(layers.UpSampling2D())
        self.bloc2conv1 = layers.TimeDistributed(
            layers.Conv2DTranspose(48, 3, padding="same", activation="relu")
        )
        self.bloc2conv2 = layers.TimeDistributed(
            layers.Conv2DTranspose(48, 3, padding="same", activation="relu")
        )

        self.bloc3upsamp = layers.TimeDistributed(layers.UpSampling2D())
        self.bloc3conv1 = layers.TimeDistributed(
            layers.Conv2DTranspose(32, 3, padding="same", activation="relu")
        )
        self.bloc3conv2 = layers.TimeDistributed(
            layers.Conv2DTranspose(32, 3, padding="same", activation="relu")
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * 8, input_shape[2] * 8, 32)

    def build(self, input_shape):
        # self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim),initializer='uniform',trainable=True)
        super(ConvLSTMDec, self).build(input_shape)

    def call(self, inputs):

        x0 = self.droplayer(inputs, training=True)

        x1 = self.bloc1upsamp(x0)
        x2 = self.bloc1conv1(x1)
        x3 = self.bloc1conv2(x2)

        x4 = self.bloc2upsamp(x3)
        x5 = self.bloc2conv1(x4)
        x6 = self.bloc2conv2(x5)

        x7 = self.bloc3upsamp(x6)
        x8 = self.bloc3conv1(x7)
        x9 = self.bloc3conv2(x8)

        return x9


def convlstm(seq_size, img_height, img_width, droprate):
    model = Sequential(
        [
            ConvLSTMEnc(droprate),
            layers.ConvLSTM2D(64, 3, padding="same", return_sequences=True),
            ConvLSTMDec(droprate),
            layers.TimeDistributed(
                layers.Conv2DTranspose(2, 1, padding="same", activation="sigmoid")
            ),
        ]
    )
    model.build((None, seq_size, img_height, img_width, 1))
    return model


if __name__ == "__main__":
    # model = Sequential([
    #    ConvLSTMEnc(0.1),
    #    layers.ConvLSTM2D(64, 3, padding='same', return_sequences=True),
    #    ConvLSTMDec(0.1),
    #    layers.TimeDistributed(layers.Conv2DTranspose(2, 1, padding='same', activation='sigmoid'))])
    # model = Unet_seq(40, 192, 192)
    model = convlstm(40, 192, 192, 0.1)
    model.summary()
