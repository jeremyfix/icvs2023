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
import os
import json
import argparse
import numpy as np

from skimage import exposure
from scipy.spatial.distance import directed_hausdorff


def Normalize(image):
    image = image.copy()
    image = image - np.amin(image)
    image = image / np.amax(image)
    return image


def HistogramCut(image, shift=0.0, cap=1.0):
    image = image.copy()
    image[image > cap] = cap
    image[image < 0] = 0.0
    image /= cap - shift
    return image


def Equalize(image):
    # Adaptive Equalization
    adv = exposure.equalize_adapthist(Normalize(image.copy()), clip_limit=0.02)
    adv = exposure.adjust_log(Normalize(adv), gain=1, inv=False)
    adv = HistogramCut(adv)
    adv = Normalize(HistogramCut(adv))
    return adv


def loadJson(json_path):
    if not os.path.exists(json_path):
        raise ValueError("Description dataset file not found.")
    jsonfile = open(json_path, "r")
    config_dic = json.load(jsonfile)
    jsonfile.close()
    return config_dic


def HD_metric(targets, preds):
    preds = np.argwhere(preds)
    targets = np.argwhere(targets)
    HD = directed_hausdorff(preds, targets)[0]
    return HD


# weights_default = "./KnotsFromContours42tree/models/20221017-113629/new_contour_model_149/epoch_149.ckpt"
def parse():
    print(
        """
    inference.py  Copyright (C) 2023  Anonymous
    This program comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions;
            """
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="./Sequences_pipeline/knots_contours/input/datasets/contours/",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="./Sequences_pipeline/knots_contours/input/datasets/knots/",
    )
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument(
        "--weights",
        type=str,
        default="./KnotsFromContours_v3/models/20221122-201224/new_contour_model_299/epoch_299.ckpt",
    )
    parser.add_argument("--species", type=str, default="sapin")
    parser.add_argument("--save_img", action="store_true")
    parser.add_argument(
        "--descriptor",
        type=str,
        default="./Sequences_pipeline/knots_contours/input/descriptors/KnotsFromContours/desc_train.json",
    )
    parser.add_argument(
        "--model", choices=["Unet", "SegNet", "ConvLSTM"], default="ConvLSTM"
    )
    return parser.parse_args()
