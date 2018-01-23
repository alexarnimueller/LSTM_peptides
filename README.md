# LSTM_peptides
## Introduction
This repository contains scripts for training a generative long short-term memory recurrent neural network on peptide 
sequences. The user can provide sets of amino acid sequences to train the model, and finally invoke sampling of 
sequences that should be similar to the training data. As such, artificial intelligence is put in charge of *de novo* design of new 
peptide sequences. The code in this repository relies on the `keras` package by Chollet and others 
(https://github.com/fchollet/keras) with `tensorflow` backend (http://tensorflow.org) as well as on 
`sklearn` (http://scikit-learn.org) and `modlamp` (https://modlamp.org).

## Content
- README.md: this file
- LSTM_peptides.py: contains the main code in the following two classes:
  - `SequenceHandler`: class that is used for reading amino acid sequences and translating them into a one-hot vector encoding. 
  - `Model`: class that generates and trains the model, can perform cross-validation and plot training and validation loss.
 - requirements.txt: requirements / package dependencies
 - LICENSE: MIT opensource license

## How To Install And Use
Clone the directory to your computer by:

``` bash
git clone https://github.com/alexarnimueller/LSTM_peptides
```

Then, install all requirements (in `requirements.txt`). In this folder, type: 

``` bash
pip install -r requirements.txt
```

Finally run the model as follows (with your own parameters provided, see the list below):

``` bash
python LSTM_peptides.py --dataset $TRAINING_DATA_FILE --name $YOUR_RUN_NAME  $FURTHER_OPTIONAL_PARAMETERS
```

#### Parameters:
- `dataset` *(default=`training_sequences_noC.csv`)*
  - file containing training data with one sequence per line
- `name` *(default=`test`)*
  - run name for all generated data; a new directory will be created with this name
- `batch_size` *(OPTIONAL, default=`128`)*
  - Batch size to use by the model.
- `epochs` *(OPTIONAL, default=`50`)*
  - Number of epochs to train the model.
- `layers` *(OPTIONAL, default=`2`)*
  - Number of LSTM layers in the model. 
- `neurons` *(OPTIONAL, default=`256`)*
  - Number of units per LSTM layer.
- `cell` *(OPTIONAL, default=`LSTM`)*
  - type of neuron to use, available: `LSTM`, `GRU`
- `dropout` *(OPTIONAL, default=`0.1`)*
  - Fraction of dropout to apply to the network. This scales with depth, so layer1 gets 1\*dropout, Layer2 2\*dropout
   etc.
- `train` *(OPTIONAL, default=`True`)*
  - Whether to train the model (`True`) or just sample from a pre-trained model (`False`).
- `valsplit` *(OPTIONAL, default=`0.2`)*
  - Fraction of the data to use for model validation. If 0, no validation is performed.
- `sample` *(OPTIONAL, default=`100`)*
  - Number of sequences to sample from the model after training.
- `temp` *(OPTIONAL, default=`1.25`)*
  - Temperature to use for sampling.
- `maxlen` *(OPTIONAL, default=`0`)*
  - Maximum sequence length allowed when sampling new sequences. If 0, the longest sequence length of the training 
  data is `maxlen`
- `startchar` *(OPTIONAL, default=`B`)*
- `lr` *(OPTIONAL, default=`0.01`)*
  - Learning rate to be used for Adam optimizer.
- `modfile` *(OPTIONAL, default=`None`)*
  - If `train=False`, a pre-trained model file needs to be provided, e.g. `modfile=./checkpoint/model_epoch_49.hdf5`.
- `cv` *(OPTIONAL, default=`None`)*
  - Folds of cross-validation to use for model validation. If `None`, no cross-validation is performed.
- `window` *(OPTIONAL, default=`0`)*
  - Size of window to use for enhancing training data by sliding-windows. If 0, all sequences are padded to the 
  length of the longest sequence in the data set.
- `step` *(OPTIONAL, default=`1`)*
  - Step size to move the sliding window or the prediction target
- `target` *(OPTIONAL, default=`all`)*
  - whether to learn all proceeding characters or just the last the single next one in sequence
- `target` *(OPTIONAL, default='all')*
  - whether to learn all proceeding characters or just the last 'one' in sequence
- `padlen` *(OPTIONAL, default=0)*
  - number of tailing padding spaces to add to the sequences. If 0, sequences are padded to the length of the longest 
  sequence in the dataset. 
- `refs`, *(OPTIONAL, default=`True`*
  - whether reference sequence sets should be generated for the analysis


### Example: training a 2-layer model with 64 neurons on new sequences for 100 epochs
``` bash
python LSTM_peptides.py --name train100 --dataset new_sequences.csv --layers 2 --neurons 64 --epochs 100
```

### Example: sampling 100 sequences from a pre-trained model
``` bash
python LSTM_peptides.py --name testsample --modfile pretrained_model/checkpoint/model_epoch_99.hdf5 --train False --sample 100
```

### Example: finetune a pre-trained model on a finetuning set for 10 epochs
``` bash
python LSTM_peptides.py --name finetune10 --dataset finetune_set.csv --modfile pretrained_model/checkpoint/model_epoch_99.hdf5 --epochs 10--train False --finetune True
```

## Cite
When using this code for any publication, please cite the following article:

A. T. Müller, J. A. Hiss, G. Schneider, "Recurrent Neural Network Model for Constructive Peptide Design" *J. Chem. Inf
. Model.* **2018**, DOI: [10.1021/acs.jcim.7b00414](http://dx.doi.org/10.1021/acs.jcim.7b00414).
