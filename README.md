# LSTM_peptides
## Introduction
This repository contains scripts for training a generative long short-term memory recurrent neural network on peptide sequences. 
The code relies on the keras package by Chollet and others (https://github.com/fchollet/keras) with TensorFlow 
backend (http://tensorflow.org) as well as on scikit-learn (http://scikit-learn.org).

## Content
- README.md: this file
- LSTM_peptides.py: contains the main code in the following two classes:
  - `SequenceHandler`: class that is used for reading amino acid sequences and translating them into a one-hot vector encoding. 
  - `Model`: class that generates and trains the model, can perform cross-validation and plot training and validation loss.
 - requirements.txt: requirements / package dependencies
 - LICENSE: MIT opensource license

## How To Use
First, install all requirements (in `requirements.txt`). In this folder, type: 

``` bash
pip install -r requirements.txt
```

Then run the model as follows:

``` bash
python LSTM_peptides.py --dataset $TRAINING_DATA_FILE --run_name $YOUR_RUN_NAME  $FURTHER_OPTIONAL_PARAMETERS
```

#### Parameters:
- `dataset`
  - file containing training data with one sequence per line
- `name`
  - run name for all generated data; a new directory will be created with this name
- `batch_size` *(OPTIONAL, default=128)*
  - Batch size to use by the model.
- `epochs` *(OPTIONAL, default=25)*
  - Number of epochs to train the model.
- `layers` *(OPTIONAL, default=2)*
  - Number of LSTM layers in the model. 
- `neurons` *(OPTIONAL, default=256)*
  - Number of units per LSTM layer.
- `valsplit` *(OPTIONAL, default=0.2)*
  - Fraction of the data to use for model validation. If 0, no validation is performed.
- `cell` *(OPTIONAL, default = 'LSTM')*
  - Recurrent cell type to use. Available: 'LSTM', 'GRU'
- `sample` *(OPTIONAL, default=100)*
  - Number of sequences to sample from the model after training.
- `temp` *(OPTIONAL, default=0.8)*
  - Temperature to use for sampling.
- `maxlen` *(OPTIONAL, default=48)*
  - Maximum sequence length allowed when sampling new sequences
- `startchar` *(OPTIONAL, default='B')*
  - Starting character to begin sampling from. Default: 'B' for 'begin' (independent)
- `dropout` *(OPTIONAL, default=0.1)*
  - Fraction of dropout to apply to the recurrent layer. This scales with depth, so layer1 gets 1\*dropout, Layer2 2\*dropout
   etc.
- `train` *(OPTIONAL, default=True)*
  - Whether to train the model (`True`) or just sample from a pre-trained model (`False`).
- `lr` *(OPTIONAL, default=0.001)*
  - Learning rate to be used for Adam optimizer.
- `l2` *(OPTIONAL, default=None)*
  - l2 regularization rate. If None, no l2 regularization is performed.
- `modfile` *(OPTIONAL, default=None)*
  - If `train=False`, a pre-trained model file needs to be provided.
- `cv` *(OPTIONAL, default=None)*
  - Folds of cross-validation to use for model validation. If 0, no cross-validation is performed.
- `window` *(OPTIONAL, default=0)*
  - Size of window to use for enhancing training data by sliding-windows. If 0, all sequences are padded to the 
  length of the longest sequence in the data set.
- `step` *(OPTIONAL, default=1)*
  - Step size to move the sliding window or the prediction target
- `target` *(OPTIONAL, default='all')*
  - whether to learn all proceeding characters or just the last 'one' in sequence
- `padlen` *(OPTIONAL, default=0)*
  - number of tailing padding spaces to add to the sequences. If 0, sequences are padded to the length of the longest 
  sequence in the dataset. 


### Example: sampling 100 sequences from a pre-trained model
``` bash
python LSTM_peptides.py --run_name testsample --modfile pretrained_model/checkpoint/model_epoch_99.hdf5 --train False --sample 100
```

### Example: training a 2-layer model on new sequences for 100 epochs
``` bash
python LSTM_peptides.py --run_name train100 --dataset new_sequences.csv --layers 2 --epochs 100
```
