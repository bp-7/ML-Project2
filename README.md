# Machine Learning Project : Text classification

Authors : Damien Ronssin, Jérémie Canoni--Meynet, Benjamin Pedrotti

## Installation

In order to be able to run the code, you will need the following libraries :
* numpy `pip install numpy`
* pandas `pip install pandas`
* tensorflow `pip install tensorflow`
* keras `pip install keras`
* symspell `pip install symspellpy`
* nltk `pip install nltk`
* gensim `pip install gensim`
* sklearn `pip install scikit-learn`

Some additional downloads are required for nltk and Symspell. Note that due to the computational cost of the whole processing, the notebooks have been executed using Google Colaboratory.

To be able to run the notebooks `LSTM.ipynb` and `LSTM_Tuning.ipynb` you will need to download the pre-trained GloVe models which can be found [here](https://nlp.stanford.edu/projects/glove/) in the section 'Download pre-trained word vectors'. The models used were in 'glove.twitter.27B.zip'.

## Usage

### Google Drive 

All the dataset and models are available in the following [Google Drive](https://drive.google.com/drive/folders/1WOL1puOGA7eccIwQgNAB0Z7JHNaeU3Ta?usp=sharing). You will be able to run the files present in this repositery with Google Colaboratory.
To execute the notebook and properly import the different modules, you will need to add the drive folder to your drive.

### Data

We did not put the original data on this repository. You will need to locate them in a folder called `data`. They are available in the Google Drive.

### Pre-processing

You will find in the file `cleaning.py` all the pre-processing methods explained in the report. This file generates three `.npy` files  with pre-processed data : the training dataset and its labels and the test dataset.
The following files are creating when running `cleaning.py`: 
* `data_train_pr_f_sl5.npy`
* `data_test_pr_f_sl5.npy`
* `labels_train_f_sl5.npy`

Concerning the generation of the raw dataset (please refer to the report), you will need to call the function `get_raw_data(path_g, full)` contained in the file `helpers.py`, where `path_g` is the Google Drive path in Google Colaboratory and `full`='f' for full dataset and 'nf' for non full dataset. This function will return the raw training dataset and its labels as well as the test dataset. 

### Hyper-parameter tuning

In the notebook classical_ML you will find the hyperparemeter tuning for the classical machine learning algorithms.

In the notebooks `CNN_Tuning.ipynb` and `LSTM_Tuning.ipynb` you will find the hyper-parameter tuning for two different deep learning architectures : convolutional network and LSTM network respectively.


### Reproducibility 

You should be able to reproduce our best results by running the script `run.py` which calls our best saved model and make a prediction with it. To run `run.py` you will need to download `best_model.hdf5` on the Google Drive as well as the content of the folder `word2vec_model`. You will need to get the original folder `data` to run it. The model has been generated with the notebook `CNN.ipynb`. You can generate similar models with this notebook but there will some differences due to stochatics processes. The different random seeds have been fixed (numpy and tensorflow), however some differences could still be present possibly because of gensim librairy. `run.py` and `CNN.ipynb` will generate a sumbmission file named `sub_best.csv`.

