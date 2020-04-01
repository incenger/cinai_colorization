# Colorization

## Library

Dependencies"

```
PyTorch
sikit-image
OpenCV2
NumPy
moviepy
tqdm
matplotlib
```

## Dataset

Put training data and testing data in  `data/train` and `data/test` respectively; otherwise, you have to specify it using flag `-p` when running `train.py` and `test.py`.

## Training 

Run the following command, the weights and loss graph will be stored in `checkpoints` and `result` respectively.

```
$ python3 train.py
```

## Testing 

Use `prepare_test.py` to prepare gray frames from ground truth frames and then run `test.py` to get the result. Result will be stored in `res` folder in the same path to every cut.

```
$ python3 prepare_test.py
$ pythone test.py
```
## Color Consistency Enhancement

Run `components.py` to perform color consistency on component of character.

```
$python3 components.py /path/to/image_folder/ /path/to/enhanced_folder/
```
