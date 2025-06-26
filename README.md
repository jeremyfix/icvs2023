
## Access to the data

For now, the data we used cannot be shared. 

All the datas for the input contour and output knots to predict have been prepared from X-Ray scans and the methodology of Khazem et al.(2023) has been applied. 

The data have been split into a training fold and a validation fold. These two folds are stored in directories `TRAIN_PATH` and `VALID_PATH`

## Dependencies

You can install the required dependencies in a virtual environment :

with `uv` :

```bash
uv venv venv
source venv/bin/activate
uv pip install scipy numpy tensorflow[and-cuda] opencv-python-headless scikit-image tqdm
```

## Training a network

### Training a ConvLSTM network

To train a ConvLSTM network on the data, you can use the `train_sequence.py` script :

```
python train_sequence.py --model_name ConvLSTM --train_path TRAIN_PATH --valid_path VALID_PATH
```

The assets of the runs are saved in the `args.output_path/models` and `args.output_path/tensorboard` directories.

### Training a non recurrent UNet or SegNet

To train a UNet or SegNet network on the data, you can use the `train_sequence.py` script. The call is similar than for training a ConvLSTM :

For a UNet : 

```
python train_sequence.py --model_name UNet --train_path TRAIN_PATH --valid_path VALID_PATH
```

For a SegNet : 

```
python train_sequence.py --model_name SegNet --train_path TRAIN_PATH --valid_path VALID_PATH
```

The assets of the runs are saved in the `args.output_path/models` and `args.output_path/tensorboard` directories.

## Running an inference 

To run inference on trees, with the specific goal of computing metrics on the test sets, we use labeled data with input in INPUT_PATH and corresponding labels in MASK_PATH. For a given model with weights saved in WEIGHTS_CKPT, you should call :

```
python inference.py --input_path INPUT_PATH --mask_path MASK_PATH --weights WEIGHT_CKPT --model {UNet, SegNet, ConvLSTM} --descriptor descriptor.json
```

The inference script can be more verbose by providing `--save_img`

The optional `species` argument, which on our case is either Fir or Spruce, is used to tag the computed metrics by a tree specie.

The outputs of this inference are saved in the `outputs` directory (can be customized by `--output` provided to the script).

For inference, the `descriptor.json` file must contain the image width and height as well as the number of slices per volume, for example with the following content :

```
seq_size: 40
input_shape: [192, 192]
```

