### Model 1: LSTM with MoT layer

* [Paper Link](https://www.aclweb.org/anthology/D16-1193/)
* Folder: Model1(LSTM_MoT)
* Weights uploaded

Set-Up:
* Install keras with Theano backend
* Prepare data. We have used 5-fold cross validation on ASAP dataset to evaluate our system. This dataset (training_set_rel3.tsv) can be downloaded from [here](https://www.kaggle.com/c/asap-aes/data). After downloading the file, put it in the [data](data) directory and create training, development and test data using `preprocess_asap.py` script

`cd data

python preprocess_asap.py -i training_set_rel3.tsv`

* Run script `train_nea.py`. You can see the list of available options by running `python train_nea.py -h`

* Training:
Though the weights are given of each prompt are given in the [output](output) directory, following command trains a model for prompt 1 in the ASAP dataset, using the training and development data from fold 0 and evaluates it.

`python train_nea.py -tr data/fold0/train.tsv -tu data/fold_0/dev.tsv -ts data/fold_0/test.tsv -p 1	# Prompt ID --emb embeddings.w2v.txt -o output_dir #Output Dir`
Here `--emb` option is to initialize the lookup table with pretrained embeddinings which is in simple Word2Vec format. To replicate our results use file `embeddings.w2v.txt` 
