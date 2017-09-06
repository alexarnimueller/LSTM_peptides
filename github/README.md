# LSTM_peptides
### Introduction
This repository contains scripts for training a generative long short-term memory recurrent neural network on peptide sequences. 
The code relies on the keras package by Chollet and others (https://github.com/fchollet/keras) and on scikit-learn (http://scikit-learn.org).

### Content
- README.md: this file
- LSTM_peptides.py: contains the main code in the following two classes:
  - `SequenceHandler`: class that is used for reading amino acid sequences and translating them into a one-hot vector encoding. 
  - `Model`: class that generates and trains the model, can perform cross-validation and plot training and validation loss.
 - requirements.txt: requirements / package dependencies
 - LICENSE: MIT opensource license

### How To Use
0) install all requirements
1) adapt the file names in the `main()` function of the LSTM_peptides.py script to your files 
2) define the model parameters in the `main()` function
3) run `python LSTM_peptides.py`
