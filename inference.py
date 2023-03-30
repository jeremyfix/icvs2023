import os
import cv2
import tensorflow as tf
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy as entropy_function

# Local import
import utils_inference
from models import SimpleConvLSTM3, SegNetSeq
from unet import Unet_seq

unet_weights = "./KnotsFromContours_Unet_2/models/20230116-133504/new_contour_model_159/epoch_159.ckpt"
segnet_weights = "./KnotsFromContours_SegNet/models/20230109-190741/new_contour_model_112/epoch_112.ckpt"


class Inference:
    def __init__(self, args):
        args = utils_inference.parse()
        info_Json = self.loadJson(args.descriptor)
        self.seq_size = info_Json["seq_size"]
        self.input_shape = (info_Json["input_shape"][0], info_Json["input_shape"][1])
        dropout = 0.2
        # df = pd.DataFrame(columns=['Specie', 'Tree', 'ID', 'Mean IoU', 'Mean Dice/F1', 'HD', 'FP', 'FN', 'TP', 'TN', "Kappa_0.1", "Kappa_0.2", "Kappa_0.3", "Kappa_0.4", "Kappa_0.4", "Kappa_0.5", "Dice_0.1", "Dice_0.2", "Dice_0.3", "Dice_0.4", "Dice_0.45", "Dice_0.5", "Dice_0.6", "Dice_0.65", "Dice_0.7"])
        df = pd.DataFrame(
            columns=[
                "Specie",
                "Tree",
                "ID",
                "HD",
                "Kappa_0.0",
                "Kappa_0.1",
                "Kappa_0.2",
                "Kappa_0.3",
                "Kappa_0.4",
                "Kappa_0.5",
                "Dice_0.0",
                "Dice_0.1",
                "Dice_0.2",
                "Dice_0.3",
                "Dice_0.4",
                "Dice_0.45",
                "Dice_0.5",
                "Dice_0.6",
                "Dice_0.65",
                "Dice_0.7",
            ]
        )
        # df = pd.read_csv("results.csv")
        self.mean_over_last = self.seq_size // 2
        if args.model == "Unet":
            model = Unet_seq(self.seq_size, self.input_shape[0], self.input_shape[1])
            model.summary()
            model.load_weights(unet_weights)
        elif args.model == "SegNet":
            model = SegNetSeq(self.seq_size, self.input_shape[0], self.input_shape[1])
            model.summary()
            model.load_weights(segnet_weights)
        else:
            model = SimpleConvLSTM3(
                self.seq_size, self.input_shape[0], self.input_shape[1], dropout
            )
            model.summary()
            model.load_weights(args.weights)
        tree_img_list = sorted(os.listdir(os.path.join(args.input_path, args.species)))
        tree_mask_list = sorted(os.listdir(os.path.join(args.mask_path, args.species)))
        # print(tree_img_list, "\n\n\n" ,tree_mask_list, tree_img_list==tree_mask_list)
        tree_path_list = [
            os.path.join(args.input_path, args.species, i) for i in tree_img_list
        ]
        tree_mask_path_list = [
            os.path.join(args.mask_path, args.species, i) for i in tree_mask_list
        ]
        # print(tree_path_list[0], "\n\n\n" ,tree_mask_path_list[0])
        count = 0
        for idx, (tree_id, tree_path) in enumerate(zip(tree_img_list, tree_path_list)):
            print(f"Processing Tree: {tree_id}")
            img_name_list = [
                img_name
                for img_name in sorted(os.listdir(tree_path))
                if img_name.split(".")[-1] == "png"
            ]
            mask_name_list = [
                mask_name
                for mask_name in sorted(
                    os.listdir(tree_path.replace("/contours", "/knots"))
                )
                if mask_name.split(".")[-1] == "png"
            ]
            img_path_list = [
                os.path.join(tree_path, img_name) for img_name in img_name_list
            ]
            mask_path_list = [
                os.path.join(tree_path.replace("/contours", "/knots"), img_name)
                for img_name in img_name_list
            ]
            # print("Inside the loop 2\n", img_path_list[:2],"\n\n", mask_path_list[:2])
            # print("Inside the loop\n", img_name_list[0],"\t", mask_name_list[0])
            num_images = int(img_name_list[-1][-10:-4])
            if len(img_name_list) < num_images * 0.95 and (len(img_name_list) < 400):
                print(
                    " * skipping tree",
                    tree_path,
                    ": not enough slices (found",
                    len(img_name_list),
                    "on",
                    num_images,
                    ")",
                )
                continue
            img_list = []
            for img_path in img_path_list:
                img = cv2.resize(
                    cv2.imread(img_path, cv2.IMREAD_UNCHANGED), self.input_shape
                )
                img = utils_inference.Equalize(img)
                img_batch = np.expand_dims(img, -1)
                img_list.append(img_batch)
            print(" * images loaded")
            mask_list = []
            for mask_path in mask_path_list:
                img = cv2.resize(
                    cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), self.input_shape
                )
                img_batch = np.expand_dims(img, -1)
                mask_list.append(img_batch)
            print(" * masks loaded")

            img_seq_list = []
            img_name_seq_list = []
            mask_seq_list = []
            mask_name_seq_list = []
            for i, j in zip(
                range(len(img_list) - self.seq_size),
                range(len(mask_list) - self.seq_size),
            ):
                img_seq_list.append(np.asarray([img_list[i : i + self.seq_size]]))
                mask_seq_list.append(np.asarray([mask_list[j : j + self.seq_size]]))
                img_name_seq_list.append(img_name_list[i : i + self.seq_size])
                mask_name_seq_list.append(img_name_list[j : j + self.seq_size])
            print(" * sequences generated")
            meaniou = tf.keras.metrics.MeanIoU(2, name="Test_meanIoU", dtype=tf.float32)
            meaniou0 = tf.keras.metrics.MeanIoU(2, name="Mean_0.0", dtype=tf.float32)
            meaniou1 = tf.keras.metrics.MeanIoU(2, name="Mean_0.1", dtype=tf.float32)
            meaniou2_1 = tf.keras.metrics.MeanIoU(2, name="Mean_0.2", dtype=tf.float32)
            meaniou2 = tf.keras.metrics.MeanIoU(2, name="Mean_0.3", dtype=tf.float32)
            meaniou3 = tf.keras.metrics.MeanIoU(2, name="Mean_0.4", dtype=tf.float32)
            meaniou4 = tf.keras.metrics.MeanIoU(2, name="Mean_0.45", dtype=tf.float32)
            meaniou5 = tf.keras.metrics.MeanIoU(2, name="Mean_0.5", dtype=tf.float32)
            meaniou6 = tf.keras.metrics.MeanIoU(2, name="Mean_0.6", dtype=tf.float32)
            meaniou7 = tf.keras.metrics.MeanIoU(2, name="Mean_0.65", dtype=tf.float32)
            meaniou8 = tf.keras.metrics.MeanIoU(2, name="Mean_0.7", dtype=tf.float32)

            FP_5 = tf.keras.metrics.FalsePositives(dtype=tf.float32)
            FN_5 = tf.keras.metrics.FalseNegatives(dtype=tf.float32)
            TN_5 = tf.keras.metrics.TrueNegatives(dtype=tf.float32)
            TP_5 = tf.keras.metrics.TruePositives(dtype=tf.float32)

            FP_0 = tf.keras.metrics.FalsePositives(dtype=tf.float32)
            FN_0 = tf.keras.metrics.FalseNegatives(dtype=tf.float32)
            TN_0 = tf.keras.metrics.TrueNegatives(dtype=tf.float32)
            TP_0 = tf.keras.metrics.TruePositives(dtype=tf.float32)

            FP_1 = tf.keras.metrics.FalsePositives(dtype=tf.float32)
            FN_1 = tf.keras.metrics.FalseNegatives(dtype=tf.float32)
            TN_1 = tf.keras.metrics.TrueNegatives(dtype=tf.float32)
            TP_1 = tf.keras.metrics.TruePositives(dtype=tf.float32)

            FP_2 = tf.keras.metrics.FalsePositives(dtype=tf.float32)
            FN_2 = tf.keras.metrics.FalseNegatives(dtype=tf.float32)
            TN_2 = tf.keras.metrics.TrueNegatives(dtype=tf.float32)
            TP_2 = tf.keras.metrics.TruePositives(dtype=tf.float32)

            FP_3 = tf.keras.metrics.FalsePositives(dtype=tf.float32)
            FN_3 = tf.keras.metrics.FalseNegatives(dtype=tf.float32)
            TN_3 = tf.keras.metrics.TrueNegatives(dtype=tf.float32)
            TP_3 = tf.keras.metrics.TruePositives(dtype=tf.float32)

            FP_4 = tf.keras.metrics.FalsePositives(dtype=tf.float32)
            FN_4 = tf.keras.metrics.FalseNegatives(dtype=tf.float32)
            TN_4 = tf.keras.metrics.TrueNegatives(dtype=tf.float32)
            TP_4 = tf.keras.metrics.TruePositives(dtype=tf.float32)

            imgs_list = []
            masks_list = []
            mean_iou = []
            mean_iou = []
            mean_iou_1 = []
            mean_iou_2_1 = []
            mean_iou_2 = []
            mean_iou_3 = []
            mean_iou_4 = []
            mean_iou_5 = []
            mean_iou_6 = []
            mean_iou_7 = []
            mean_iou_8 = []
            false_positives = []
            false_negatives = []
            true_positives = []
            true_negatives = []
            pred_dict = {name: [] for name in img_name_list}
            gt_dict = {name: [] for name in img_name_list}
            for img_seq, img_name_seq, mask_seq, mask_name_seq in tqdm(
                zip(
                    img_seq_list,
                    img_name_seq_list,
                    mask_seq_list,
                    mask_name_seq_list,
                ),
                total=len(img_seq_list),
            ):
                count += 1
                pred_seq = model(img_seq).numpy()
                # Clear backend session to avoid memory leakage
                tf.keras.backend.clear_session()
                # print(pred_seq, "\n", mask_seq, "\n Shape Pred/Mask", pred_seq.shape, mask_seq.shape, pred_seq.max() * 255.0, pred_seq.min(), mask_seq.max(), mask_seq.min())
                y_true = mask_seq / 255.0
                pred_copy = np.copy(pred_seq)
                pred_0 = np.copy((pred_seq > 0.0).astype(np.float))
                pred_1 = np.copy((pred_seq > 0.1).astype(np.float))
                pred_2 = np.copy((pred_seq > 0.2).astype(np.float))
                pred_3 = np.copy((pred_seq > 0.3).astype(np.float))
                pred_4 = np.copy((pred_seq > 0.4).astype(np.float))
                pred_45 = np.copy((pred_seq > 0.45).astype(np.float))
                pred_5 = np.copy((pred_seq > 0.5).astype(np.float))
                pred_6 = np.copy((pred_seq > 0.6).astype(np.float))
                pred_65 = np.copy((pred_seq > 0.65).astype(np.float))
                pred_7 = np.copy((pred_seq > 0.7).astype(np.float))

                # TP FN TN FP for differents thresholds
                FP_5.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_5[:, :, :, :, 0]), tf.int32),
                )
                FN_5.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_5[:, :, :, :, 0]), tf.int32),
                )
                TP_5.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_5[:, :, :, :, 0]), tf.int32),
                )
                TN_5.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_5[:, :, :, :, 0]), tf.int32),
                )

                FP_4.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_4[:, :, :, :, 0]), tf.int32),
                )
                FN_4.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_4[:, :, :, :, 0]), tf.int32),
                )
                TP_4.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_4[:, :, :, :, 0]), tf.int32),
                )
                TN_4.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_4[:, :, :, :, 0]), tf.int32),
                )

                FP_3.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_3[:, :, :, :, 0]), tf.int32),
                )
                FN_3.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_3[:, :, :, :, 0]), tf.int32),
                )
                TP_3.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_3[:, :, :, :, 0]), tf.int32),
                )
                TN_3.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_3[:, :, :, :, 0]), tf.int32),
                )

                FP_2.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_2[:, :, :, :, 0]), tf.int32),
                )
                FN_2.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_2[:, :, :, :, 0]), tf.int32),
                )
                TP_2.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_2[:, :, :, :, 0]), tf.int32),
                )
                TN_2.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_2[:, :, :, :, 0]), tf.int32),
                )

                FP_1.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_1[:, :, :, :, 0]), tf.int32),
                )
                FN_1.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_1[:, :, :, :, 0]), tf.int32),
                )
                TP_1.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_1[:, :, :, :, 0]), tf.int32),
                )
                TN_1.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_1[:, :, :, :, 0]), tf.int32),
                )

                FP_0.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_0[:, :, :, :, 0]), tf.int32),
                )
                FN_0.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_0[:, :, :, :, 0]), tf.int32),
                )
                TP_0.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_0[:, :, :, :, 0]), tf.int32),
                )
                TN_0.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_0[:, :, :, :, 0]), tf.int32),
                )

                meaniou.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_copy[:, :, :, :, 0]), tf.int32),
                )
                meaniou0.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_0[:, :, :, :, 0]), tf.int32),
                )
                meaniou1.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_1[:, :, :, :, 0]), tf.int32),
                )
                meaniou2_1.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_2[:, :, :, :, 0]), tf.int32),
                )
                meaniou2.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_3[:, :, :, :, 0]), tf.int32),
                )
                meaniou3.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_4[:, :, :, :, 0]), tf.int32),
                )
                meaniou4.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_45[:, :, :, :, 0]), tf.int32),
                )
                meaniou5.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_5[:, :, :, :, 0]), tf.int32),
                )
                meaniou6.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_6[:, :, :, :, 0]), tf.int32),
                )
                meaniou7.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_65[:, :, :, :, 0]), tf.int32),
                )
                meaniou8.update_state(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32),
                    tf.cast(tf.math.round(pred_7[:, :, :, :, 0]), tf.int32),
                )

                fc_0 = (
                    (
                        (TN_0.result().numpy() + FN_0.result().numpy())
                        * (TN_0.result().numpy() + FP_0.result().numpy())
                    )
                    + (
                        (FP_0.result().numpy() + TP_0.result().numpy())
                        * (FN_0.result().numpy() + TP_0.result().numpy())
                    )
                ) / (
                    TP_0.result().numpy()
                    + TN_0.result().numpy()
                    + FN_0.result().numpy()
                    + FP_0.result().numpy()
                )

                fc_1 = (
                    (
                        (TN_1.result().numpy() + FN_1.result().numpy())
                        * (TN_1.result().numpy() + FP_1.result().numpy())
                    )
                    + (
                        (FP_1.result().numpy() + TP_1.result().numpy())
                        * (FN_1.result().numpy() + TP_1.result().numpy())
                    )
                ) / (
                    TP_1.result().numpy()
                    + TN_1.result().numpy()
                    + FN_1.result().numpy()
                    + FP_1.result().numpy()
                )

                fc_2 = (
                    (
                        (TN_2.result().numpy() + FN_2.result().numpy())
                        * (TN_2.result().numpy() + FP_2.result().numpy())
                    )
                    + (
                        (FP_2.result().numpy() + TP_2.result().numpy())
                        * (FN_2.result().numpy() + TP_2.result().numpy())
                    )
                ) / (
                    TP_2.result().numpy()
                    + TN_2.result().numpy()
                    + FN_2.result().numpy()
                    + FP_2.result().numpy()
                )

                fc_3 = (
                    (
                        (TN_3.result().numpy() + FN_3.result().numpy())
                        * (TN_3.result().numpy() + FP_3.result().numpy())
                    )
                    + (
                        (FP_3.result().numpy() + TP_3.result().numpy())
                        * (FN_3.result().numpy() + TP_3.result().numpy())
                    )
                ) / (
                    TP_3.result().numpy()
                    + TN_3.result().numpy()
                    + FN_3.result().numpy()
                    + FP_3.result().numpy()
                )

                fc_4 = (
                    (
                        (TN_4.result().numpy() + FN_4.result().numpy())
                        * (TN_4.result().numpy() + FP_4.result().numpy())
                    )
                    + (
                        (FP_4.result().numpy() + TP_4.result().numpy())
                        * (FN_4.result().numpy() + TP_4.result().numpy())
                    )
                ) / (
                    TP_4.result().numpy()
                    + TN_4.result().numpy()
                    + FN_4.result().numpy()
                    + FP_4.result().numpy()
                )

                fc_5 = (
                    (
                        (TN_5.result().numpy() + FN_5.result().numpy())
                        * (TN_5.result().numpy() + FP_5.result().numpy())
                    )
                    + (
                        (FP_5.result().numpy() + TP_5.result().numpy())
                        * (FN_5.result().numpy() + TP_5.result().numpy())
                    )
                ) / (
                    TP_5.result().numpy()
                    + TN_5.result().numpy()
                    + FN_5.result().numpy()
                    + FP_5.result().numpy()
                )

                Kappa_0 = ((TP_0.result().numpy() + TN_0.result().numpy()) - fc_0) / (
                    (
                        TP_0.result().numpy()
                        + TN_0.result().numpy()
                        + FN_0.result().numpy()
                        + FP_0.result().numpy()
                    )
                    - fc_0
                )
                Kappa_1 = ((TP_1.result().numpy() + TN_1.result().numpy()) - fc_1) / (
                    (
                        TP_1.result().numpy()
                        + TN_1.result().numpy()
                        + FN_1.result().numpy()
                        + FP_1.result().numpy()
                    )
                    - fc_1
                )
                Kappa_2 = ((TP_2.result().numpy() + TN_2.result().numpy()) - fc_2) / (
                    (
                        TP_2.result().numpy()
                        + TN_2.result().numpy()
                        + FN_2.result().numpy()
                        + FP_2.result().numpy()
                    )
                    - fc_2
                )
                Kappa_3 = ((TP_3.result().numpy() + TN_3.result().numpy()) - fc_3) / (
                    (
                        TP_3.result().numpy()
                        + TN_3.result().numpy()
                        + FN_3.result().numpy()
                        + FP_3.result().numpy()
                    )
                    - fc_3
                )
                Kappa_4 = ((TP_4.result().numpy() + TN_4.result().numpy()) - fc_4) / (
                    (
                        TP_4.result().numpy()
                        + TN_4.result().numpy()
                        + FN_4.result().numpy()
                        + FP_4.result().numpy()
                    )
                    - fc_4
                )
                Kappa_5 = ((TP_5.result().numpy() + TN_5.result().numpy()) - fc_5) / (
                    (
                        TP_5.result().numpy()
                        + TN_5.result().numpy()
                        + FN_5.result().numpy()
                        + FP_5.result().numpy()
                    )
                    - fc_5
                )

                df.loc[count, "Kappa_0.0"] = Kappa_0
                df.loc[count, "Kappa_0.1"] = Kappa_1
                df.loc[count, "Kappa_0.2"] = Kappa_2
                df.loc[count, "Kappa_0.3"] = Kappa_3
                df.loc[count, "Kappa_0.4"] = Kappa_4
                df.loc[count, "Kappa_0.5"] = Kappa_5
                df.loc[count, "Specie"] = args.species
                df.loc[count, "Tree"] = tree_id
                df.loc[count, "ID"] = img_name_seq
                # df.loc[count, "TP"] = TP.result().numpy()
                # df.loc[count, "TN"] = TN.result().numpy()
                # df.loc[count, "FP"] = FP.result().numpy()
                # df.loc[count, "FN"] = FN.result().numpy()
                # df.loc[count, "Kappa"] = metrics.cohen_kappa_score(tf.cast(y_true[:, :, :, :, 0], tf.int32).numpy(), tf.cast(tf.math.round(pred_copy[:, :, :, :, 0]), tf.int32).numpy())
                # df.loc[count, "Mean IoU"] = meaniou.result().numpy()*100
                # df.loc[count, "Mean Dice/F1"] = ((2*(meaniou.result().numpy())) / (meaniou.result().numpy() + 1))*100

                df.loc[count, "Dice_0.0"] = (
                    (2 * (meaniou0.result().numpy())) / (meaniou0.result().numpy() + 1)
                ) * 100
                df.loc[count, "Dice_0.1"] = (
                    (2 * (meaniou1.result().numpy())) / (meaniou1.result().numpy() + 1)
                ) * 100
                df.loc[count, "Dice_0.2"] = (
                    (2 * (meaniou2_1.result().numpy()))
                    / (meaniou2_1.result().numpy() + 1)
                ) * 100
                df.loc[count, "Dice_0.3"] = (
                    (2 * (meaniou2.result().numpy())) / (meaniou2.result().numpy() + 1)
                ) * 100
                df.loc[count, "Dice_0.4"] = (
                    (2 * (meaniou3.result().numpy())) / (meaniou3.result().numpy() + 1)
                ) * 100
                df.loc[count, "Dice_0.45"] = (
                    (2 * (meaniou4.result().numpy())) / (meaniou4.result().numpy() + 1)
                ) * 100
                df.loc[count, "Dice_0.5"] = (
                    (2 * (meaniou5.result().numpy())) / (meaniou5.result().numpy() + 1)
                ) * 100
                df.loc[count, "Dice_0.6"] = (
                    (2 * (meaniou6.result().numpy())) / (meaniou6.result().numpy() + 1)
                ) * 100
                df.loc[count, "Dice_0.65"] = (
                    (2 * (meaniou7.result().numpy())) / (meaniou7.result().numpy() + 1)
                ) * 100
                df.loc[count, "Dice_0.7"] = (
                    (2 * (meaniou8.result().numpy())) / (meaniou8.result().numpy() + 1)
                ) * 100
                df.loc[count, "HD"] = utils_inference.HD_metric(
                    tf.cast(y_true[:, :, :, :, 0], tf.int32).numpy(),
                    tf.cast(tf.math.round(pred_copy[:, :, :, :, 0]), tf.int32).numpy(),
                )

                # print(f"Tree: {tree_id}, ID: {img_name_seq}, F1: {((2*(meaniou.result().numpy())) / (meaniou.result().numpy() + 1))*100}\n")
                # print("MeanIoU: ", meaniou.result().numpy()*100, "| mean Dice/F1:", ((2*(meaniou.result().numpy())) / (meaniou.result().numpy() + 1))*100)
                mean_iou.append(meaniou.result().numpy() * 100)
                mean_iou_1.append(meaniou1.result().numpy() * 100)
                mean_iou_2_1.append(meaniou2_1.result().numpy() * 100)
                mean_iou_2.append(meaniou2.result().numpy() * 100)
                mean_iou_3.append(meaniou3.result().numpy() * 100)
                mean_iou_4.append(meaniou4.result().numpy() * 100)
                mean_iou_5.append(meaniou5.result().numpy() * 100)
                mean_iou_6.append(meaniou6.result().numpy() * 100)
                mean_iou_7.append(meaniou7.result().numpy() * 100)
                mean_iou_8.append(meaniou8.result().numpy() * 100)
                false_positives.append(FP_0.result().numpy())
                false_negatives.append(FN_0.result().numpy())
                true_positives.append(TP_0.result().numpy())
                true_negatives.append(TN_0.result().numpy())
                # print("Images", img_name_seq, "\nMasks", mask_name_seq, "\n")
                for i, name in enumerate(
                    img_name_seq[-self.mean_over_last :], self.mean_over_last
                ):
                    pred_dict[name].append(pred_seq[0, i])

            thresh_ = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.65, 0.7]
            if args.save_img:
                os.makedirs(
                    os.path.join(args.output, args.species, tree_id, "preds"),
                    exist_ok=True,
                )
                os.makedirs(
                    os.path.join(args.output, args.species, tree_id, "preds_thresh"),
                    exist_ok=True,
                )
                os.makedirs(
                    os.path.join(args.output, args.species, tree_id, "entropy"),
                    exist_ok=True,
                )
                os.makedirs(
                    os.path.join(args.output, args.species, tree_id, "variance"),
                    exist_ok=True,
                )
                for i in thresh_:
                    os.makedirs(
                        os.path.join(args.output, args.species, tree_id, f"preds_{i}"),
                        exist_ok=True,
                    )
                for name in pred_dict.keys():
                    result = np.asarray(pred_dict[name])
                    if result.shape[0] > 10:
                        preds = np.mean(result[:, :, :, 1], axis=0)
                        variance = np.std(result[:, :, :, 1], axis=0)
                        entropy = entropy_function(result[:, :, :, 1], axis=0)
                        # cv2.imwrite(os.path.join(args.output, args.species, tree_id, 'preds', name), (preds*255).astype(np.uint8))
                        # for thresh in np.arange(0.3, 0.8, 0.1):
                        #    cv2.imwrite(os.path.join(args.output, args.species, tree_id, f'preds_thresh_{thresh}', name), ((preds>thresh)*255).astype(np.uint8))

                        # cv2.imwrite(os.path.join(args.output, args.species, tree_id, 'variance', name), (variance*255).astype(np.uint8))
                        # cv2.imwrite(os.path.join(args.output, args.species, tree_id, 'entropy', name), (entropy*255).astype(np.uint8))
                        for i in thresh_:
                            cv2.imwrite(
                                os.path.join(
                                    args.output,
                                    args.species,
                                    tree_id,
                                    f"preds_{i}",
                                    name,
                                ),
                                ((preds > i) * 255).astype(np.uint8),
                            )

                print(" * images saved")
        os.makedirs("./output/", exist_ok=True)
        df.to_csv(
            f"./output/{args.model}_{args.species}_metrics_thresholds.csv", index=False
        )
        # print(len(mean_iou), np.array(mean_iou).mean(), "Best:", np.array(mean_iou).max())
        print(df.head())
        print(" * processing done")

    def loadJson(self, json_path):
        if not os.path.exists(json_path):
            raise ValueError("Description dataset file not found.")
        jsonfile = open(json_path, "r")
        config_dic = json.load(jsonfile)
        jsonfile.close()
        return config_dic


if __name__ == "__main__":
    args = utils_inference.parse()
    model = Inference(args)
