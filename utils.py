import numpy as np
from subprocess import Popen, PIPE
import tensorflow.compat.v1 as tf1
import tensorflow as tf


def encode_gif(frames, fps):
    h, w, c = frames[0].shape
    pxfmt = {1: "gray", 3: "rgb24"}[3]
    cmd = " ".join(
        [
            "ffmpeg -y -f rawvideo -vcodec rawvideo",
            "-r "
            + str("%.02f" % fps)
            + " -s "
            + str(w)
            + "x"
            + str(h)
            + " -pix_fmt "
            + pxfmt
            + " -i - -filter_complex",
            "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
            "-r " + str("%.02f" % fps) + " -f gif -",
        ]
    )
    proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
        out, err = proc.communicate()
        if proc.returncode:
            raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
        del proc
    return out


def video_summary(name, video, step=None, fps=10):
    name = name if isinstance(name, str) else name.decode("utf-8")
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)  # .numpy()
        summary.value.add(tag=name + "/gif", image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print("GIF summaries require ffmpeg in $PATH.", e)
    frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
    tf.summary.image(name + "/grid", frames.astype(np.uint8), step)


# result_array = []
# array_orig = tensor_orig.numpy()
# array_pred = tensor_pred.numpy()
# for i in range(len(array_orig)):
#   heatmap = array_pred[i,:,:,0].copy()
#   heatmap /= np.max(heatmap)
#   heatmap = np.uint8(-255 * (heatmap-1))
#   im_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#   im_orig = np.concatenate((array_orig[i].copy(),)*3, axis=-1).astype(np.uint8)
#   result_array.append(cv2.addWeighted(im_orig,0.6,im_color,0.4,0))
