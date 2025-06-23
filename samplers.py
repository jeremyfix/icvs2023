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
import numpy as np
import json
import cv2
import os
import random
import skimage as ski
import tensorflow as tf


def HistogramCut(img, shift=0.0, cap=1.0):
    # restrict the histogram to a rante between 'shift' and 'cap'
    # should be between 0. and 1., with shift < cap
    img = img.copy()
    img[img > cap] = cap

    img -= shift
    img[img < 0] = 0.0

    img /= cap - shift

    return img


def Normalize(img):
    # spread an image histogram over the bit range
    # print (NP.amin(img), NP.amax(img))
    img = img.copy()

    img = img - np.amin(img)
    img = img / np.amax(img)
    # img = img % 1

    return img


def Equalize(img):
    # Performs adaptative equalization
    ada = Normalize(img.copy())
    ada = ski.exposure.equalize_adapthist(ada, clip_limit=0.02)
    ada = Normalize(ada)
    ada = ski.exposure.adjust_log(ada, gain=1, inv=False)
    ada = HistogramCut(ada)
    ada = Normalize(ada)
    return ada


class SimpleDataset:
    def __init__(self, ds_description_path):
        self._loadInfo(ds_description_path)

    def getDataset(self):
        generator = self._generator
        return tf.data.Dataset.from_generator(
            generator,
            args=[],
            output_types=(tf.float32, tf.uint8),
            output_shapes=(
                tf.TensorShape(self._info["input_shape"] + [1]),
                tf.TensorShape(self._info["output_shape"] + [1]),
            ),
        )

    def _generator(self):
        img_list = self._img_list
        lbl_list = self._lbl_list
        for i in range(self.num_samples):
            img = np.expand_dims(
                cv2.resize(
                    cv2.imread(img_list[i], 0), dsize=tuple(self._info["input_shape"])
                ),
                -1,
            )
            lbl = np.expand_dims(
                cv2.resize(
                    cv2.imread(lbl_list[i], 0), dsize=tuple(self._info["input_shape"])
                ),
                -1,
            )
            yield (img, lbl)

    def _loadInfo(self, path):
        self._info = self._loadDict(path)
        self._img_list = self._loadList(self._info["img_list_path"])
        self._lbl_list = self._loadList(self._info["lbl_list_path"])
        self.num_samples = self._info["num_samples"]
        print(self._info)

    def _loadList(self, path):
        if not os.path.exists(path):
            raise ValueError("File " + path + " not found. Aborting.")
        content = []
        file_ = open(path, "r")
        for line in file_:
            content.append(line.strip())
        file_.close()
        return content

    def _loadDict(self, path):
        # print("Path : ", path)
        if not os.path.exists(path):
            raise ValueError("Dataset description file not found. Aborting.")

        file_ = open(path, "r")
        dic = json.load(file_)
        file_.close()
        return dic


class SequenceDataset(SimpleDataset):
    def __init__(self, ds_description_path):
        super().__init__(ds_description_path)

    def _generate_sequences(self):
        self.img_ram = {}
        self.lbl_ram = {}
        self._matched_seq = []
        for img_path, lbl_path in zip(self._img_list, self._lbl_list):
            # Extract and format data
            image_names = [
                name
                for name in sorted(os.listdir(img_path))
                if name.split(".")[-1] == "png"
            ]
            label_names = [
                name
                for name in sorted(os.listdir(lbl_path))
                if name.split(".")[-1] == "png"
            ]
            image_pathes = [os.path.join(img_path, name) for name in image_names]
            label_pathes = [os.path.join(lbl_path, name) for name in label_names]
            label_nums = [int(name.split(".")[0]) for name in label_names]

            pot_lbl_nums = [
                label_nums[i : i + self.seq_size]
                for i in range(len(label_nums) - self.seq_size)
            ]
            pot_lbl_names = [
                label_names[i : i + self.seq_size]
                for i in range(len(label_names) - self.seq_size)
            ]
            pot_lbl_pathes = [
                label_pathes[i : i + self.seq_size]
                for i in range(len(label_pathes) - self.seq_size)
            ]
            pot_img_names = [
                image_names[i : i + self.seq_size]
                for i in range(len(image_names) - self.seq_size)
            ]
            pot_img_pathes = [
                image_pathes[i : i + self.seq_size]
                for i in range(len(image_pathes) - self.seq_size)
            ]

            # Ensure label sequence continuity
            valid_seq = [
                ((seq[-1] - seq[0]) == (self.seq_size - 1)) for seq in pot_lbl_nums
            ]
            out = [
                [pot_lbl_names[i], pot_lbl_pathes[i]]
                for i, cond in enumerate(valid_seq)
                if cond
            ]
            pot_lbl_names, pot_lbl_pathes = list(map(list, zip(*out)))
            # Match label and image sequences
            for i, img_name_seq in enumerate(pot_img_names):
                for j, lbl_name_seq in enumerate(pot_lbl_names):
                    if lbl_name_seq == img_name_seq:
                        self._matched_seq.append([pot_img_pathes[i], pot_lbl_pathes[j]])

        self.num_samples = len(self._matched_seq)

        for sequence_pair in self._matched_seq:
            for img_path, lbl_path in zip(*sequence_pair):
                if img_path not in self.img_ram.keys():
                    self.img_ram[img_path] = Equalize(
                        cv2.resize(
                            cv2.imread(img_path, -1), tuple(self._info["input_shape"])
                        )
                    )
                if lbl_path not in self.lbl_ram.keys():
                    self.lbl_ram[lbl_path] = cv2.resize(
                        cv2.imread(lbl_path, -1), tuple(self._info["input_shape"])
                    )

    def _generator(self):
        random.shuffle(self._matched_seq)
        for pair in self._matched_seq:
            img_seq, lbl_seq = self._augment_sequence_pair(pair)
            yield (img_seq, lbl_seq)

    def _augment_sequence_pair(self, sequence_pair):
        img_seq = []
        lbl_seq = []
        # 7 rotations
        rot_idx = int(np.random.rand() * 7)
        # 3 flips
        flip_idx = int(np.random.rand() * 3)
        # Load element per element and apply augmentation
        for img_path, lbl_path in zip(*sequence_pair):
            # Load and resize
            img = self.img_ram[
                img_path
            ].copy()  # (cv2.imread(img_path,-1), tuple(self._info['input_shape']))
            lbl = self.lbl_ram[
                lbl_path
            ].copy()  # .resize(cv2.imread(lbl_path,-1), tuple(self._info['output_shape']))
            # Apply rotation
            filler = int(img[0, 0])
            img = cv2.warpAffine(
                img,
                self.rot_mat_img[rot_idx],
                tuple(self._info["input_shape"]),
                borderValue=filler,
            )
            filler = int(lbl[0, 0])
            lbl = cv2.warpAffine(
                lbl,
                self.rot_mat_lbl[rot_idx],
                tuple(self._info["output_shape"]),
                borderValue=filler,
            )
            # Apply flip
            img = cv2.flip(img, flip_idx)
            lbl = cv2.flip(lbl, flip_idx)
            img_seq.append(img)
            lbl_seq.append(lbl)

        if self.reversed_mode:
            img_seq.reverse()
            lbl_seq.reverse()

        img_seq = np.expand_dims(np.array(img_seq), -1)
        lbl_seq = np.expand_dims(np.array(lbl_seq), -1)
        return [img_seq, lbl_seq]

    def _loadInfo(self, path):
        self._info = self._loadDict(path)
        # TODO: why this fix was necessary ??
        # Is this version of the code the latest ? Otherwise _generate_sequences() will fail
        # because the img_list and lbl_list are empty
        # But the super class also has a _loadInfo method, so it is not clear why this is necessary
        # self._img_list = self._info["img_list_path"]
        # self._lbl_list = self._info["lbl_list_path"]
        self._img_list = self._loadList(self._info["img_list_path"])
        self._lbl_list = self._loadList(self._info["lbl_list_path"])
        self.seq_size = self._info["seq_size"]
        self.reversed_mode = self._info["reversed_mode"]
        self._generate_sequences()
        self._ready_augmentation()
        print(self._info)

    def _ready_augmentation(self):
        self.rot_mat_img = []
        self.rot_mat_lbl = []
        for angle in np.linspace(0, 360, 8)[:-1]:
            self.rot_mat_img.append(
                cv2.getRotationMatrix2D(
                    (
                        self._info["input_shape"][0] / 2,
                        self._info["input_shape"][1] / 2,
                    ),
                    angle,
                    1,
                )
            )
            self.rot_mat_lbl.append(
                cv2.getRotationMatrix2D(
                    (
                        self._info["output_shape"][0] / 2,
                        self._info["output_shape"][1] / 2,
                    ),
                    angle,
                    1,
                )
            )

    def getDataset(self):
        generator = self._generator
        return tf.data.Dataset.from_generator(
            generator,
            args=[],
            output_types=(tf.float32, tf.uint8),
            output_shapes=(
                tf.TensorShape(
                    [self._info["seq_size"]] + self._info["input_shape"] + [1]
                ),
                [self._info["seq_size"]]
                + tf.TensorShape(self._info["output_shape"] + [1]),
            ),
        )
