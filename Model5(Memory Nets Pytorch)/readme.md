# Memory Networks Pytorch

## Usage
```
# Train on essay set 1
python train.py --set_id 1
```
There are serval flags within train.py.
```
  --gpu_id        GPU_ID
  --set_id        SET_ID         essay set id, 1 <= id <= 8.
  --emb_size      EMB_SIZE       Embedding size for sentences.
  --token_num     TOKEN_NUM      The number of token in glove (6, 42).
  --feature_size  FEATURE_SIZE   Feature size.
  --epochs        EPOCHS         Number of epochs to train for.
  --test_freq     TEST_FREQ      Evaluate and print results every x epochs.
  --hops          HOPS           Number of hops in the Memory Network.
  --lr            LR             Learning rate.
  --batch_size    BATCH_SIZE     Batch size for training.
  --l2_lambda     L2_LAMBDA      Lambda for l2 loss.
  --num_samples   NUM_SAMPLES    Number of samples selected as memories for each score.
  --epsilon       EPSILON        Epsilon value for Adam Optimizer.
  --max_grad_norm MAX_GRAD_NORM  Clip gradients to this norm.
  --keep_prob     KEEP_PROB      Keep probability for dropout.
```
Better performance can be get by tuning hyper-parameters.

## Glove
Pre-trained word embeddings are used in this model. You can download `glove_42B_300d` from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/).