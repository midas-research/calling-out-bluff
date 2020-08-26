import sys 
argslist = list(sys.argv)[1:]
print(argslist)

import tensorflow as tf
import data_utils as data_utils 
from qwk import quadratic_weighted_kappa
import time
import os
import sys
import pandas as pd
import numpy as np

graph = tf.get_default_graph()

essay_set_id = argslist[0]
adv_file = argslist[1]
print("PromptID is: ", essay_set_id)
print(type(essay_set_id))
print("Adversarial File: ", adv_file)

# essay_set_id = 3
num_tokens = 42
embedding_size = 300
num_samples = 1
is_regression = True

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

training_path = "training_set_rel3.tsv"
essay_list, resolved_scores, essay_id = data_utils.load_training_data(training_path, int(essay_set_id))
print(len(resolved_scores))

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
print("train max sent size: ", max_sent_size)
mean_sent_size = int(np.mean(sent_size_list))
E = data_utils.vectorize_data(essay_list, word_idx, max_sent_size)

a = True
if a == True:
#     print("Testcase: ", testcase)
    testing_path = adv_file
    apple = True
    if apple==True:
        df_essay = pd.read_csv(testing_path, engine='python')
        print(len(df_essay))
    essay_list_test, essay_id_test = data_utils.load_testcase_data(df_essay, essay_set_id)
    E_test = data_utils.vectorize_data(essay_list_test, word_idx, max_sent_size)
    test_batch_size = 15

    testE = []
    test_scores = []
    test_essay_id = []
    for test_index in range(len(essay_id_test)):
        testE.append(E_test[test_index])
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
    print("training_data: ", n_train)
    print("ADV Testcase: ", adv_file, n_test)

    memory = []
    for i in score_range:
        for j in range(num_samples):
            if i in train_scores:
                score_idx = train_scores.index(i)
                essay = trainE.pop(score_idx)
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

    #CHANGE DIRECTORIES and rename checkpoints for each prompt
    saver = tf.train.import_meta_graph("Model5(MemoryNets)/runs/essay_set_3_cv_1_Jun_04_2020_23:32:11/checkpoints-11830.meta")  
    with tf.Session() as sess:
        saver.restore(sess,"Model5(MemoryNets)/runs/essay_set_3_cv_1_Jun_04_2020_23:32:11/checkpoints-11830")
        query = graph.get_tensor_by_name("input/question:0")
        memory_key = graph.get_tensor_by_name("input/memory_key:0")
        keep_prob = graph.get_tensor_by_name("input/keep_prob:0")
        for op in graph.get_operations():
            name2 = op.name
            if name2.startswith("input"): 
                print(name2)
        output = graph.get_tensor_by_name("prediction/predict_op:0")

        test_preds = []
        for start in range(0, n_test, test_batch_size):
            end = min(n_test, start+test_batch_size)
            batched_memory = [memory] * (end-start)
            query_list = []
            for elem2 in testE[start:end]:
                if (len(elem2) == max_sent_size):
                    query_list.append(elem2)
                else:
                    elem2 = elem2[:max_sent_size]
                    query_list.append(elem2)
            preds = sess.run(output, feed_dict={query: query_list, memory_key:batched_memory, keep_prob:1})
            if is_regression:
                preds = np.clip(np.round(preds), min_score, max_score)
            predsF = preds.astype('float32') 
            if type(predsF) is np.float32:
                test_preds = np.append(test_preds, predsF)
            else:
                preds = preds.astype('int32')
                try:
                    preds2 = preds.tolist()
                except ValueError:
                    print(adv_file, e)
                    pass
                preds2 = preds.tolist()
                for ite in range(len(preds2)):
                    test_preds = np.append(test_preds, preds2[ite])
            pred_dict = {'pred_score':test_preds}
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv('memoryNets_pred_'+adv_file, index=None)
