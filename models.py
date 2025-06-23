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
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras import layers
import tensorflow as tf

# Local import
from ConvLSTM_utils import ConvLSTMencode, ConvLSTMdecode
from SegNetSeq_utils import SegNetencode, SegNetdecode

# from CBAM import CBAMblock


def ConvLSTM(seq_size, img_height, img_width, droprate):

    model = Sequential(
        [
            layers.TimeDistributed(
                ConvLSTMencode(droprate),
                input_shape=(seq_size, img_height, img_width, 1),
            ),
            layers.ConvLSTM2D(64, 3, padding="same", return_sequences=True),
            layers.TimeDistributed(ConvLSTMdecode(droprate)),
            layers.TimeDistributed(
                layers.Conv2DTranspose(1, 1, padding="same", activation="sigmoid")
            ),
        ]
    )

    return model


def SegNetSeq(seq_size, img_height, img_width):
    model = Sequential(
        [
            layers.TimeDistributed(
                SegNetencode(), input_shape=(seq_size, img_height, img_width, 1)
            ),
            layers.TimeDistributed(SegNetdecode()),
            layers.TimeDistributed(
                layers.Conv2DTranspose(1, 1, padding="same", activation="sigmoid")
            ),
        ]
    )
    return model


@tf.function
def check(x, y):
    return tf.cond(tf.greater(x, y), lambda: True, lambda: False)


def CBAMmodel(input_shape):
    img_height, img_width = input_shape
    inputs = Input((img_height, img_width, 1))
    outputs = layers.experimental.preprocessing.Rescaling(
        1.0 / 255, input_shape=(img_height, img_width, 1)
    )(inputs)
    outputs = layers.Conv2D(64, 3, padding="same", activation="relu")(outputs)
    outputs = layers.Conv2D(64, 3, padding="same", activation="relu")(outputs)
    outputs = layers.MaxPooling2D()(outputs)

    outputs = CBAMblock(chans=1)(outputs)

    outputs = layers.UpSampling2D()(outputs)
    outputs = layers.Conv2DTranspose(64, 3, padding="same", activation="relu")(outputs)
    outputs = layers.Conv2DTranspose(64, 3, padding="same", activation="relu")(outputs)

    # projection
    outputs = layers.Conv2DTranspose(3, 1, padding="same", activation="sigmoid")(
        outputs
    )

    model = Model(inputs=[inputs], outputs=[outputs], name="CA")
    return model


class CBAM(tf.keras.Model):
    def __init__(self, input_shape, chans=1):
        super(CBAM, self).__init__()
        self.rescale = layers.experimental.preprocessing.Rescaling(
            1.0 / 255, input_shape=(input_shape[0], input_shape[1], 1)
        )
        self.layenc1 = layers.Conv2D(64, 3, padding="same", activation="relu")
        self.layenc2 = layers.Conv2D(64, 3, padding="same", activation="relu")
        # self.cbamlayenc = CBAMblock(chans=64)
        self.layenc3 = layers.MaxPooling2D()
        self.cbamlaymiddle = CBAMblock(chans=chans)
        self.laydec1 = layers.UpSampling2D()
        self.laydec2 = layers.Conv2DTranspose(64, 3, padding="same", activation="relu")
        self.laydec3 = layers.Conv2DTranspose(64, 3, padding="same", activation="relu")
        self.layproj = layers.Conv2DTranspose(
            3, 1, padding="same", activation="sigmoid"
        )

    def call(self, inputs):
        x = self.rescale(inputs)
        x = self.layenc1(x)
        x = self.layenc2(x)
        # x = self.cbamlayenc(x)
        x = self.layenc3(x)
        x = self.cbamlaymiddle(x)
        x = self.laydec1(x)
        x = self.laydec2(x)
        x = self.laydec3(x)
        x = self.layproj(x)
        return x


if __name__ == "__main__":
    from unet import Unet_seq

    model = SegNetSeq(40, 192, 192)  # seq_size, img_height, img_width,
    model_3 = Unet_seq(40, 192, 192)
    # model.summary()
    # model_3.summary()
