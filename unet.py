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
from tensorflow.keras import layers, Sequential, Input, Model


def Unet_seq(seq_size, img_height, img_width):
    inputs = Input((seq_size, img_height, img_width, 1))
    # Bloc 1
    conv_1 = Sequential(
        [
            layers.TimeDistributed(
                layers.Conv2D(8, 3, padding="same", activation="relu"),
                input_shape=(seq_size, img_height, img_width, 1),
            ),
            layers.TimeDistributed(
                layers.Conv2D(8, 3, padding="same", activation="relu")
            ),
        ]
    )(inputs)
    bn_1 = layers.TimeDistributed(layers.BatchNormalization())(conv_1)
    pool_1 = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(bn_1)

    # Bloc 2
    conv_2 = Sequential(
        [
            layers.TimeDistributed(
                layers.Conv2D(16, 3, padding="same", activation="relu")
            ),
            layers.TimeDistributed(
                layers.Conv2D(16, 3, padding="same", activation="relu")
            ),
        ]
    )(pool_1)
    bn_2 = layers.TimeDistributed(layers.BatchNormalization())(conv_2)
    pool_2 = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(bn_2)

    # Bloc 3
    conv_3 = Sequential(
        [
            layers.TimeDistributed(
                layers.Conv2D(32, 3, padding="same", activation="relu")
            ),
            layers.TimeDistributed(
                layers.Conv2D(32, 3, padding="same", activation="relu")
            ),
        ]
    )(pool_2)
    bn_3 = layers.TimeDistributed(layers.BatchNormalization())(conv_3)
    pool_3 = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(bn_3)

    # Bloc 4
    conv_4 = Sequential(
        [
            layers.TimeDistributed(
                layers.Conv2D(64, 3, padding="same", activation="relu")
            ),
            layers.TimeDistributed(
                layers.Conv2D(64, 3, padding="same", activation="relu")
            ),
        ]
    )(pool_3)

    # Bloc 5
    up1 = layers.TimeDistributed(layers.UpSampling2D())(conv_4)
    conv_5 = layers.concatenate([up1, conv_3])
    conv_5 = Sequential(
        [
            layers.TimeDistributed(
                layers.Conv2D(64, 3, padding="same", activation="relu")
            ),
            layers.TimeDistributed(
                layers.Conv2D(64, 3, padding="same", activation="relu")
            ),
        ]
    )(conv_5)

    # Bloc 6
    up2 = layers.TimeDistributed(layers.UpSampling2D())(conv_5)
    conv_6 = layers.concatenate([up2, conv_2])
    conv_6 = Sequential(
        [
            layers.TimeDistributed(
                layers.Conv2D(32, 3, padding="same", activation="relu")
            ),
            layers.TimeDistributed(
                layers.Conv2D(32, 3, padding="same", activation="relu")
            ),
        ]
    )(conv_6)

    # Bloc 7
    up3 = layers.TimeDistributed(layers.UpSampling2D())(conv_6)
    conv_7 = layers.concatenate([up3, conv_1])
    conv_7 = Sequential(
        [
            layers.TimeDistributed(
                layers.Conv2D(16, 3, padding="same", activation="relu")
            ),
            layers.TimeDistributed(
                layers.Conv2D(16, 3, padding="same", activation="relu")
            ),
        ]
    )(conv_7)

    output = layers.TimeDistributed(
        layers.Conv2DTranspose(1, 1, padding="same", activation="sigmoid")
    )(conv_7)
    # output = layers.TimeDistributed(layers.Conv2D(2, 1, padding="same", activation='softmax'))(conv_7)
    model = Model(inputs=[inputs], outputs=[output], name="Unet")
    return model


if __name__ == "__main__":
    model = Unet_seq(40, 192, 192)
    model.summary()
