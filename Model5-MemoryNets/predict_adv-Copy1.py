import tensorflow as tf
import data_utils
from qwk import quadratic_weighted_kappa
import time
import os
import sys
import pandas as pd
import numpy as np

graph = tf.get_default_graph()

essay_set_id = 1
num_tokens = 42
embedding_size = 300
num_samples = 1
is_regression = False

early_stop_count = 0
max_step_count = 10
is_regression = False
gated_addressing = False
# essay_set_id = 1
batch_size = 15
embedding_size = 300
feature_size = 100
l2_lambda = 0.3
hops = 3
reader = 'bow' # gru may not work
epochs = 200
num_samples = 1
num_tokens = 42
test_batch_size = batch_size
random_state = 0

training_path = 'training_set_rel3.tsv'
essay_list, resolved_scores, essay_id = data_utils.load_training_data(training_path, essay_set_id)

max_score = max(resolved_scores)
min_score = min(resolved_scores)
if essay_set_id == 7:
    min_score, max_score = 0, 30
elif essay_set_id == 8:
    min_score, max_score = 0, 60

print("Max Score", max_score)
print("Min Score", min_score)
score_range = range(min_score, max_score+1)
# load glove
word_idx, word2vec = data_utils.load_glove(num_tokens, dim=embedding_size)

vocab_size = len(word_idx) + 1
print("vocab size", vocab_size)
# stat info on data set

sent_size_list = list(map(len, [essay for essay in essay_list]))
max_sent_size = max(sent_size_list)
mean_sent_size = int(np.mean(sent_size_list))
E = data_utils.vectorize_data(essay_list, word_idx, max_sent_size)

testing_path = 'aes_data/essay1/fold_0/test.txt'
essay_list_test, resolved_scores_test, essay_id_test = data_utils.load_testing_data(testing_path, essay_set_id)

test_batch_size = 15

testE = []
test_scores = []
test_essay_id = []
for test_index in range(len(essay_id_test)):
    testE.append(E[test_index])
    test_scores.append(resolved_scores_test[test_index])
    test_essay_id.append(essay_id_test[test_index])

trainE = []
train_scores = []
train_essay_id = []
for train_index in range(len(essay_id)):
    trainE.append(E[train_index])
    train_scores.append(resolved_scores[train_index])
    train_essay_id.append(essay_id[train_index])

n_train = len(trainE)    
n_test = len(testE)

memory = []
for i in score_range:
    for j in range(num_samples):
        if i in train_scores:
            score_idx = train_scores.index(i)
#             score = train_scores.pop(score_idx)
            essay = trainE.pop(score_idx)
#             sent_size = sent_size_list.pop(score_idx)
            memory.append(essay)

def test_step(e, m):
    feed_dict = {
        query: e,
        memory_key: m,
        keep_prob: 1
        #model.w_placeholder: word2vec
    }
    preds = sess.run(output, feed_dict)
    if is_regression:
        preds = np.clip(np.round(preds), min_score, max_score)
        return preds
    else:
        return preds
        
saver = tf.train.import_meta_graph("runs/essay_set_1_cv_1_Mar_25_2020_19:43:40/checkpoints-2820.meta")  
with tf.Session() as sess:
    saver.restore(sess,"runs/essay_set_1_cv_1_Mar_25_2020_19:43:40/checkpoints-2820")
    query = graph.get_tensor_by_name("input/question:0")
    memory_key = graph.get_tensor_by_name("input/memory_key:0")
    keep_prob = graph.get_tensor_by_name("input/keep_prob:0")
#     for op in graph.get_operations():
#         print(op.name)
    output = graph.get_tensor_by_name("prediction/predict_op:0")
#     output=tf.get_collection('predict_op:0')
    
    test_preds = []
    for start in range(0, n_test, test_batch_size):
        end = min(n_test, start+test_batch_size)
        print("Start: ", start, "End: ", end)
#         print("Memory", memory)
        batched_memory = [memory] * (end-start)
#         print("Batched_memory", batched_memory)
        preds = sess.run(output, feed_dict={query: testE[start:end], memory_key:batched_memory, keep_prob:1})
#         preds = test_step(testE[start:end], batched_memory)
#         print("Preds", preds)
#         preds = preds.tolist()
        predsF = preds.astype('float32') 
        if type(predsF) is np.float32:
            test_preds = np.append(test_preds, predsF)
        else:
            preds = preds.astype('int32')
            preds2 = preds.tolist()
#             print("Preds2",preds2) 
            for ite in range(len(preds2)):
#                 ite2 = ite.astype(numpy.int32)
#                 print("Ite", type(ite))
#                 print("pred ite", preds2[ite])
                test_preds = np.append(test_preds, preds2[ite])
#                 np.concatenate(test_preds, preds2[ite])
#                 test_preds.append(preds2[ite])
        if not is_regression:
#             test_preds = np.add(test_preds, min_score)
            #test_kappp_score = kappa(test_scores, test_preds, 'quadratic')
#             test_kappp_score = quadratic_weighted_kappa(test_scores, test_preds, min_score, max_score)
#             print(test_kappp_score)
#             stat_dict = {'pred_score': test_preds}
            stat_dict = {'essay_id': test_essay_id, 'org_score': test_scores, 'pred_score': test_preds}
            pred_dict = {'pred_score':test_preds}
    
    test_kappp_score = quadratic_weighted_kappa(test_scores, test_preds, min_score, max_score)
    print(test_kappp_score)
    stat_df = pd.DataFrame(stat_dict)
    pred_df = pd.DataFrame(pred_dict)
    print(stat_df)
    stat_df.to_csv('statistics/stat_pred_prompt1.csv')
    pred_df.to_csv('statistics/pred_prompt1.csv')