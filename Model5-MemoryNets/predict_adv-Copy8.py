import tensorflow as tf
import data_utils
from qwk import quadratic_weighted_kappa
import time
import os
import sys
import pandas as pd
import numpy as np

graph = tf.get_default_graph()

essay_set_id = 8
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

training_path = 'training_set_rel3.tsv'
essay_list, resolved_scores, essay_id = data_utils.load_training_data(training_path, essay_set_id)
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

testcases_path = 'AES_FinalTestcases/prompt8/'
# print(testcases_path)
testcases_files = os.listdir(testcases_path)
print("Number of files", len(testcases_files))
testcases_files = sorted(testcases_files)
testcases_files = ["babel_prompt8.csv", "entities_beg_unbound_8.csv"]
# te
# testcases_files.remove("babel_prompt8.csv")
# testcases_files.remove("test7.csv")
# testcases_files.remove("svo_triplets_all_prompt8.csv")
# testcases_files.remove("svo_triplets_random_prompt8.csv")
# testcases_files.remove("entities_beg_unbound_1.csv")
# testcases_files.remove("entities_end_bound_1.csv")
# testcases_files.remove("entities_end_unbound_1.csv")
# testcases_files.remove("entities_mid_bound_1.csv")
# testcases_files.remove("entities_mid_unbound_1.csv")
print(testcases_files)

# testcases_files = ['disfluency_1.csv', 'incorrect_grammar_1.csv']
for testcase in testcases_files:
    print("Testcase: ", testcase)
    testing_path = testcases_path+testcase
    apple = True
#     if testcase == 'incorrect_1.csv':
#         df_essay = pd.read_csv(testing_path, engine='python')
# #         df_essay = df_essay[:0]
#         df_score = pd.read_csv('statistics/pred_prompt1.csv')
#         df_essay['rating'] = df_score['pred_score']
#         df_essay['text2'] = df_essay['text']
#         df_essay = df_essay[['text', 'rating']]
#         print(df_essay.head())
#     if (testcase =='test8.csv' or 'babel_prompt1.csv'):
#         continue
#         df_essay = pd.read_csv(testing_path, engine='python')
#         df_essay = df_essay.iloc[:,0]
#         print(df_essay.head())
    if apple==True:
#         print("Before: ", df_essay)
        df_essay = pd.read_csv(testing_path, engine='python')
#         print("Before: ", df_essay)
#         df_essay[:-1]
#         print("After: ", df_essay)
        print(len(df_essay))
#         df_score = pd.read_csv('predicted_scores/org_scores/prompt2_org.csv')
#         print(len(df_score))
#         df_essay['rating'] = df_score['score']
#         print(df_essay.head())
    
#     testing_path = 'aes_data/essay1/fold_0/test.txt'
    essay_list_test, essay_id_test = data_utils.load_testcase_data(df_essay, essay_set_id)
    
#     sent_size_list_test = list(map(len, [essay2 for essay2 in essay_list_test]))
#     max_sent_size_test = max(sent_size_list_test)
#     print("test max sent size: ", max_sent_size_test)
#     mean_sent_size_test = int(np.mean(sent_size_list_test))

    E_test = data_utils.vectorize_data(essay_list_test, word_idx, max_sent_size)
#     print(essay_list_test)
    test_batch_size = 15

    testE = []
    test_scores = []
    test_essay_id = []
    for test_index in range(len(essay_id_test)):
        testE.append(E_test[test_index])
#         print(testE)
#         test_scores.append(resolved_scores_test[test_index])
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
    print("Testcase: ", testcase, n_test)

    memory = []
    for i in score_range:
        for j in range(num_samples):
            if i in train_scores:
                score_idx = train_scores.index(i)
    #             score = train_scores.pop(score_idx)
                essay = trainE.pop(score_idx)
#                 print("essay", essay)
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
    
    
    saver = tf.train.import_meta_graph("runs/essay_set_8_cv_1_Jun_25_2020_00:48:03/checkpoints-2700.meta")  
    with tf.Session() as sess:
        saver.restore(sess,"runs/essay_set_8_cv_1_Jun_25_2020_00:48:03/checkpoints-2700")
        query = graph.get_tensor_by_name("input/question:0")
        print(query.get_shape().as_list())
        memory_key = graph.get_tensor_by_name("input/memory_key:0")
#         print(memory.get_shape().as_list())
        keep_prob = graph.get_tensor_by_name("input/keep_prob:0")
        #for op in graph.get_operations():
            #print(op.name)
        output = graph.get_tensor_by_name("prediction/predict_op:0")
#         y0 = sess.run([y_])
    #     output=tf.get_collection('predict_op:0')

        test_preds = []
        for start in range(0, n_test, test_batch_size):
#             print("E1")
            end = min(n_test, start+test_batch_size)
            print("Start: ", start, "End: ", end)
#             print("Memory", memory)
            batched_memory = [memory] * (end-start)
#             batched_memory = []
            total = end-start
            for i in range(0, total):
                batched_memory[i] = batched_memory[i][:33]
#                 print(i, '   Done')
#             print(len(batched_memory))
#             batched_memory[0] = batched_memory[0][:33]
#             print(len(batched_memory[0]))
#             print(batched_memory[14][:33])
#             print(len(batched_memory[0][0]))
#             print("Query: ", testE[start:end])
#             print("E2")
#             print(testE[start:end])
            query_list = []
            for elem2 in testE[start:end]:
#                 print(len(elem2))
                if (len(elem2) == 983):
                    query_list.append(elem2)
                else:
                    elem2 = elem2[:983]
                    query_list.append(elem2)
#                     elem2 = [int(x) for x in elem2]
#                     print(len(elem2))
#                 else:
#                     print(len(elem2))
#                     elem2 = [int(x) for x in elem2]
            for ab in query_list:
                print(len(ab))
#              print(len(ab))
#             if (start == 15 and end == 30):
#                 print("############################")
#                 print(testE[15:30])
#             for elem in batched_memory:
#                 print(len(elem))
#             print(batched_memory)
            #print(len(query_list))
            preds = sess.run(output, feed_dict={query: query_list, memory_key:batched_memory, keep_prob:1})
            print("Preds", preds)
            print("type: ", type(preds))
            
            if is_regression:
                preds = np.clip(np.round(preds), min_score, max_score)
                print(preds)
#                 return preds
#             else:
#                 continue
    #         preds = test_step(testE[start:end], batched_memory)
    #         print("Preds", preds)
    #         preds = preds.tolist()
            predsF = preds.astype('float32') 
            if type(predsF) is np.float32:
                print("Here1")
                test_preds = np.append(test_preds, predsF)
            else:
                preds = preds.astype('int32')
                print("Here2")
#                 preds2 = []
#                 for i in range(0, len(preds)):    
#                     print(preds[i])
#                     preds2.append(preds[i])
                try:
                    preds2 = preds.tolist()
                except ValueError:
                    print(testcase, e)
                    pass
#                     preds2 = []
#                     for i in range(0, len(preds)):    
#                         print(preds[i])
#                         pred2.append(preds[i])
                print("type here: ", type(preds))
                preds2 = preds.tolist()
                print("Preds2",preds2)
#                 print("pred_list", pred_list)
                for ite in range(len(preds2)):
#                     print("Done")
    #                 ite2 = ite.astype(numpy.int32)
    #                 print("Ite", type(ite))
    #                 print("pred ite", preds2[ite])
                    test_preds = np.append(test_preds, preds2[ite])
    #                 np.concatenate(test_preds, preds2[ite])
    #                 test_preds.append(preds2[ite])
#             if not is_regression:
    #             test_preds = np.add(test_preds, min_score)
                #test_kappp_score = kappa(test_scores, test_preds, 'quadratic')
    #             test_kappp_score = quadratic_weighted_kappa(test_scores, test_preds, min_score, max_score)
    #             print(test_kappp_score)
    #             stat_dict = {'pred_score': test_preds}
#             stat_dict = {'essay_id': test_essay_id, 'org_score': test_scores, 'pred_score': test_preds}
            pred_dict = {'pred_score':test_preds}

#         test_kappp_score = quadratic_weighted_kappa(test_scores, test_preds, min_score, max_score)
#         print(test_kappp_score)
#         stat_df = pd.DataFrame(stat_dict)
        pred_df = pd.DataFrame(pred_dict)
#         print(stat_df)
#         stat_df.to_csv('predicted_scores/prompt1/stats_'+testcase, index=None)
        pred_df.to_csv('predicted_scores/prompt8/pred_'+testcase, index=None)
