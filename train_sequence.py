import os
import argparse
import numpy as np
import tensorflow as tf
import datetime

from samplers import SequenceDataset
from utils import video_summary
from ConvLSTM_utils import convlstm
from models import SegNetSeq
from unet import Unet_seq


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", choices=["ConvLSTM", "UNet", "SegNet"], required=True
    )
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--val_path", type=str)
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--weights", type=str, default=None)
    return parser.parse_args()


args = parse()

TRN_SIS = SequenceDataset(args.train_path)
VAL_SIS = SequenceDataset(args.val_path)

if args.model_name == "ConvLSTM":
    model = convlstm(
        TRN_SIS._info["seq_size"],
        TRN_SIS._info["input_shape"][0],
        TRN_SIS._info["input_shape"][1],
        0.2,
    )
    batch_size = 10
elif args.model_name == "UNet":
    model = Unet_seq(
        TRN_SIS._info["seq_size"],
        TRN_SIS._info["input_shape"][0],
        TRN_SIS._info["input_shape"][1],
    )
    batch_size = 4
elif args.model_name == "SegNet":
    model = SegNetSeq(
        TRN_SIS._info["seq_size"],
        TRN_SIS._info["input_shape"][0],
        TRN_SIS._info["input_shape"][1],
    )
    batch_size = 4

if args.weights is not None:
    model.load_weights(args.weights)

train_ds = TRN_SIS.getDataset()
val_ds = VAL_SIS.getDataset()

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

model.summary()
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)
train_accuracy = tf.keras.metrics.BinaryAccuracy("train_accuracy")
val_loss = tf.keras.metrics.Mean("val_loss", dtype=tf.float32)
val_accuracy = tf.keras.metrics.BinaryAccuracy("val_accuracy")
train_meaniou = tf.keras.metrics.MeanIoU(2, name="train_meanIoU", dtype=tf.float32)
val_meaniou = tf.keras.metrics.MeanIoU(2, name="val_meanIoU", dtype=tf.float32)

# class_weights = [0.1, 0.9]


def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        y_train = y_train / 255
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(tf.reduce_mean(loss))
    train_accuracy(y_train, predictions)
    pred_copy = np.copy(predictions)
    train_meaniou.update_state(
        tf.cast(y_train[:, :, :, :, 0], tf.int32),
        tf.cast(tf.math.round(pred_copy[:, :, :, :, 0]), tf.int32),
    )


def val_step(model, x_val, y_val):
    y_val = y_val / 255
    predictions = model(x_val)
    loss = loss_object(y_val, predictions)
    val_loss(loss)
    val_accuracy(y_val, predictions)
    pred_copy = np.copy(predictions)
    val_meaniou.update_state(
        tf.cast(y_val[:, :, :, :, 0], tf.int32),
        tf.cast(tf.math.round(pred_copy[:, :, :, :, 0]), tf.int32),
    )


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
try:
    os.mkdir(os.path.join(args.ouput_path, "models"))
except:
    pass
try:
    os.mkdir(os.path.join(args.output_path, "models", current_time))
except:
    pass


train_log_dir = os.path.join(args.output_path, "tensorboard/" + current_time + "/train")
test_log_dir = os.path.join(args.output_path, "tensorboard/" + current_time + "/test")
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(test_log_dir)

max_steps = TRN_SIS.num_samples // batch_size
max_steps_prct = max_steps // 10
EPOCHS = 150

max_outputs = 4
for epoch in range(EPOCHS):
    for step, (x_train, y_train) in enumerate(train_ds.batch(batch_size)):
        train_step(model, optimizer, x_train, y_train)
        if step % max_steps_prct == 0:
            template = "Epoch {} {}%, Loss: {}, Pixel-Accuracy: {}%, MeanIoU: {}%, Mean F1: {}%"
            print(
                template.format(
                    epoch + 1,
                    int(100 * step // max_steps),
                    train_loss.result(),
                    train_accuracy.result() * 100,
                    train_meaniou.result() * 100,
                    (
                        (2 * (train_meaniou.result().numpy()))
                        / (train_meaniou.result().numpy() + 1)
                    )
                    * 100,
                )
            )
            with train_summary_writer.as_default():
                video_summary(
                    "prediction",
                    model(x_train).numpy()[:3],
                    step=epoch * max_steps + step,
                )
                video_summary(
                    "ground_truth", y_train.numpy()[:3], step=epoch * max_steps + step
                )
                overlay = tf.dtypes.cast(
                    (x_train[:3] - np.min(x_train[:3]))
                    / (np.max(x_train[:3]) - np.min(x_train[:3]))
                    * 255,
                    tf.uint8,
                ).numpy()
                overlay = np.concatenate((overlay,) * 3, axis=-1)  # convert to rgb
                overlay[:, :, :, :, 1] = y_train.numpy()[:3, :, :, :, 0]
                video_summary(
                    "groundtruth_with_overlay", overlay, step=epoch * max_steps + step
                )
                overlay = tf.dtypes.cast(
                    (x_train[:3] - np.min(x_train[:3]))
                    / (np.max(x_train[:3]) - np.min(x_train[:3]))
                    * 255,
                    tf.uint8,
                ).numpy()
                overlay = np.concatenate((overlay,) * 3, axis=-1)  # convert to rgb
                overlay[:, :, :, :, 1] = np.max(
                    tf.dtypes.cast(255 * model(x_train[:3]), tf.uint8).numpy(), axis=-1
                )
                video_summary(
                    "prediction_with_overlay", overlay, step=epoch * max_steps + step
                )
                tf.summary.scalar(
                    "loss", train_loss.result(), step=epoch * max_steps + step
                )
                tf.summary.scalar(
                    "miou", train_meaniou.result(), step=epoch * max_steps + step
                )
                tf.summary.scalar(
                    "accuracy", train_accuracy.result(), step=epoch * max_steps + step
                )
                tf.summary.flush()

                train_loss.reset_states()
                train_accuracy.reset_states()
                train_meaniou.reset_states()

    for (x_val, y_val) in val_ds.batch(batch_size):
        val_step(model, x_val, y_val)
    with val_summary_writer.as_default():
        tf.summary.scalar("loss", val_loss.result(), step=epoch * max_steps + step)
        tf.summary.scalar(
            "accuracy", val_accuracy.result(), step=epoch * max_steps + step
        )
        tf.summary.scalar("miou", val_meaniou.result(), step=epoch * max_steps + step)
        video_summary(
            "test_prediction", model(x_val).numpy(), step=epoch * max_steps + step
        )
        video_summary("test_ground_truth", y_val.numpy(), step=epoch * max_steps + step)
        overlay = tf.dtypes.cast(x_val / 256, tf.uint8).numpy()
        overlay = np.concatenate((overlay,) * 3, axis=-1)  # convert to rgb
        overlay[:, :, :, :, 1] = y_val.numpy()[:, :, :, :, 0]
        video_summary(
            "test_groundtruth_with_overlay", overlay, step=epoch * max_steps + step
        )
        overlay = tf.dtypes.cast(x_val / 256, tf.uint8).numpy()
        overlay = np.concatenate((overlay,) * 3, axis=-1)  # convert to rgb
        overlay[:, :, :, :, 1] = np.max(
            tf.dtypes.cast(255 * model(x_val), tf.uint8).numpy(), axis=-1
        )
        # print(np.max(overlay),np.max(np.max(tf.dtypes.cast(255*model(x_train), tf.uint8).numpy(), axis=-1)))
        video_summary(
            "test_prediction_with_overlay", overlay, step=epoch * max_steps + step
        )

    template = "Epoch {}, Loss: {}, Pixel-Accuracy: {}, Test Loss: {}, Test Pixel-Accuracy: {}, Train MeanIoU: {}, Val MeanIoU:{}, Val MeanF1: {}"
    print(
        template.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result() * 100,
            val_loss.result(),
            val_accuracy.result() * 100,
            train_meaniou.result() * 100,
            val_meaniou.result() * 100,
            ((2 * (val_meaniou.result().numpy())) / (val_meaniou.result().numpy() + 1))
            * 100,
        )
    )

    # Reset metrics every epoch
    val_loss.reset_states()
    val_accuracy.reset_states()
    val_meaniou.reset_states()
    train_meaniou.reset_states()
    model.save(
        os.path.join(
            args.output_path, "models", current_time, "new_contour_model_" + str(epoch)
        )
    )
    checkpoint_path = os.path.join(
        args.output_path,
        "models",
        current_time,
        "new_contour_model_" + str(epoch) + "/epoch_" + str(epoch) + ".ckpt",
    )
    model.save_weights(checkpoint_path)
