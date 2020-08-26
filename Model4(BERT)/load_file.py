import xgboost as xgb
import os
import numpy as np
import yaml
import json
import pandas as pd
from util import *
from predict import *
dataset_score_range = {
    1: (2, 12, 1783),
    2: (1, 6, 1800),
    3: (0, 3, 1726),
    4: (0, 3, 1772),
    5: (0, 4, 1805),
    6: (0, 4, 1800),
    7: (0, 30, 1569),
    8: (0, 60, 723)
}
with open("config/sys_conf.yaml", encoding="utf-8") as conf_reader:
    sys_conf = yaml.load(conf_reader.read())
with open("config/train_conf.json", "r") as cr:
    train_conf = json.load(cr)
with open("config/doc_conf.json", "r") as cr:
    doc_conf = json.load(cr)    

saved_model_dir = "overall_score_new/"
tfrecord_file_path = os.path.join(sys_conf["data_dir"], "asap_dataset_prompt.tfrecord")

#Make feature files for each test
files = os.listdir("AES_FinalTestcases/prompt5/")
print(files)
print(len(files))
# files = ["svo_triplets_all_prompt5.csv", "svo_triplets_random_prompt5.csv", "test4.csv"]
files.remove("svo_triplets_all_prompt5.csv")
files.remove("svo_triplets_random_prompt5.csv")
files.remove("test4.csv")
# files = ['uf_beg_bound_2.csv', 'uf_beg_unbound_2.csv', 'uf_end_bound_2.csv', 'uf_end_unbound_2.csv', 'uf_mid_bound_2.csv', 'uf_mid_unbound_2.csv', 'ut_beg_bound_2.csv', 'ut_beg_unbound_2.csv', 'ut_end_bound_2.csv', 'ut_end_unbound_2.csv', 'ut_mid_bound_2.csv', 'ut_mid_unbound_2.csv', 'wiki_beg_bound_2.csv', 'wiki_beg_unbound_2.csv', 'wiki_end_bound_2.csv', 'wiki_end_unbound_2.csv', 'wiki_mid_bound_2.csv', 'wiki_mid_unbound_2.csv', 'wiki_rel_beg_bound_2.csv', 'wiki_rel_beg_unbound_2.csv', 'wiki_rel_end_bound_2.csv', 'wiki_rel_end_unbound_2.csv', 'wiki_rel_mid_bound_2.csv', 'wiki_rel_mid_unbound_2.csv', 'wiki_topic_beg_bound_2.csv', 'wiki_topic_beg_unbound_2.csv', 'wiki_topic_end_bound_2.csv', 'wiki_topic_end_unbound_2.csv', 'wiki_topic_mid_bound_2.csv', 'wiki_topic_mid_unbound_2.csv']
# apple = jkfefkqfhq

# files = "AES_FinalTestcases/prompt1/svo_triplets_random_prompt1.csv"
# apple = True
# if apple == True:
for testcases in files:
    testcases_name = testcases[:-4]
    print(testcases_name)
#     apple = lkewflkqfnqf
    xgboost_adv_file_path = "feature_files/prompt5/asap_xgboost_"+testcases_name+".npz"
#     xgboost_adv_file_path = "feature_files/prompt1/asap_xgboost_new.npz"
    adv_file = "AES_FinalTestcases/prompt5/"+testcases
#     adv_file = "AES_FinalTestcases/prompt1/svo_triplets_random_prompt1.csv"
    def read_asap_dataset():
        # asap数据集的相关参数，配置，这里做全局变量使用，方便下面三个score predictor调用
        asap_csv_file_path = os.path.join(sys_conf["data_dir"], "prompt5.csv")
    #     adv_file_apth = adv_file
        print("ASAP_CSV_FILE_PATH", asap_csv_file_path)
        if not os.path.exists(asap_csv_file_path):
            raise ValueError("asap_file_path is invalid.")
        asap_dataset = pd.read_csv(asap_csv_file_path, encoding='utf-8')
        adv_dataset = pd.read_csv(adv_file, encoding='utf-8')
#         adv_dataset = adv_dataset[[2]]
        adv_dataset.columns = ['essay']
#         print(adv_dataset.head())
        adv_dataset.insert(0, 'ID', range(1, 1 + len(adv_dataset)))
        if (adv_dataset.iloc[0]["essay"] == "text"):
            adv_dataset = pd.read_csv(adv_file, encoding='utf-8')
            adv_dataset.insert(0, 'ID', range(1, 1 + len(adv_dataset)))
        adv_dataset = adv_dataset[["ID", "essay"]]
#         print(len(adv_dataset))
#         print(adv_dataset.head())
#         if adv_dataset["ID"][0] == '1':
#             print(adv_dataset[1][1])
#         apple = fqfnjhfk
    #     asap_dataset = pd.read_csv(asap_csv_file_path, encoding='ISO-8859-1')
        articles_id = list(adv_dataset["ID"])
        articles_set = list(asap_dataset["essay_set"])
        domain1_score = asap_dataset["domain1_score"]
        handmark_scores = dict(zip(articles_id, domain1_score))
        set_ids = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: []
        }
        for i in range(len(articles_id)):
            set_ids[articles_set[i]].append(articles_id[i])

        return articles_id, articles_set, domain1_score

    def read_asap_dataset_correspond():
        # asap数据集的相关参数，配置，这里做全局变量使用，方便下面三个score predictor调用
        asap_csv_file_path = os.path.join(sys_conf["data_dir"], "prompt5.csv")
    #     adv_file_apth = adv_file
    #     print("ASAP_CSV_FILE_PATH", asap_csv_file_path)
        if not os.path.exists(asap_csv_file_path):
            raise ValueError("asap_file_path is invalid.")
        asap_dataset = pd.read_csv(asap_csv_file_path, encoding='utf-8')
        adv_dataset = pd.read_csv(adv_file, encoding='utf-8', header=None)
        adv_dataset.insert(0, 'ID', range(1, 1 + len(adv_dataset)))
        #     print(adv_dataset.head())
    #     asap_dataset = pd.read_csv(asap_csv_file_path, encoding='ISO-8859-1')
        articles_id = list(adv_dataset["ID"])
        articles_set = list(asap_dataset["essay_set"])
        domain1_score = asap_dataset["domain1_score"]
        handmark_scores = dict(zip(articles_id, domain1_score))
        set_ids = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: []
        }
        for i in range(len(articles_id)):
            set_ids[articles_set[i]].append(articles_id[i])

    #     return articles_id, articles_set, domain1_score
        return articles_id, articles_set, set_ids, handmark_scores

    def generate_xgboost_train_set(articles_id, articles_set, domain1_scores, train_set_gec_result_path, train_set_saved_path):
        """Generate xgboost training data set based on the result of the training set gec

        Args:
            articles_id: list of training set article ids
            articles_set: list of training set articles
            domain1_scores: the manually labeled scores of the articles in the training set, because the asap dataset calls this score domain1_scores
            train_set_gec_result_path: The path of the result file generated by the gec engine in the training set article, the file format is a line corresponding to the gec result of an article.
            train_set_saved_path: save as npz file type, save path of npz file

        Returns: None.

        """
        dataset_gec_path = train_set_gec_result_path
        dataset_xgboost_train_file = train_set_saved_path

        # normalized_scores
        handmark_scores = dict(zip(articles_id, domain1_scores))

        # normalized_orgin_scores
        handmark_normalized_scores = {}
        for key, value in handmark_scores.items():
            article_set_id = articles_set[articles_id.index(key)]
            min_value = dataset_score_range[article_set_id][0]
            max_value = dataset_score_range[article_set_id][1]
            normalize_value = (value - min_value) / (max_value - min_value)
            handmark_normalized_scores[key] = normalize_value

        features = {}

    #     gec_output= []
    #     .insert(0, 'ID', range(1, 1 + len(adv_dataset)))
        count = 0
        with open(dataset_gec_path, encoding="ISO-8859-1") as fr:
            for line in fr:
                count +=1
    #             print(line[0], line[3])
    #             id = line[0]
    #             print(count, line)
    #             line_split = line.split(",")
    #             print(line_split)
                id = count
                print("ID", id)
    #             print(line_split[0], line_split[2])
    #             id = int(line_split[0].strip())
                gec_output = line.strip()
    #             gec_output = gec_output.encode("utf-8")
    #             print(gec_output)
                #feats = FeatureExtractor()
                #feats.initialize_dictionaries(gec_output)
                #features[id] = feats.gen_feats(gec_output) 
                features[id] = Document(gec_output).features
    #             gec_output = []
    #             print("DONE ID: ", id)
        # TODO(Jiawei): may have bugs if basic_scores的key和features的key不一样
    #     for key, value in handmark_normalized_scores.items():
    #         if key in features:
    #             features[key].append(value)

        np.savez(dataset_xgboost_train_file, features=features)
        print("Done")

    articles_id, articles_set, domain1_score = read_asap_dataset()
    generate_xgboost_train_set(articles_id, articles_set, domain1_score, adv_file, xgboost_adv_file_path)


    #Load Saved model
    xgb_rg = xgb.XGBRegressor(n_estimators=5000, learning_rate=0.001, max_depth=6, gamma=0.05,
                                      objective="reg:logistic")
    xgb_rg.load_model(os.path.join(saved_model_dir, "osp3.xgboost"))

    #Score Range
    max_value = 4
    min_value = 0
    # print("MAX AND MIN", max_value, min_value)

    #Load feature file of adversary testcase
    features_adv = np.load(xgboost_adv_file_path, allow_pickle=True)["features"][()]
    print("ADV", features_adv)

    #Generate correspond_adv_id_set
    articles_id, articles_set, set_ids, handmark_scores = read_asap_dataset_correspond()
    permutation_adv_ids = np.random.permutation(set_ids[train_conf["prompt_id"]])
    # print(permutation_adv_ids)
    correspond_adv_id_set = permutation_adv_ids

    #basic_scores, coherent scores, prompt scores
    # basic_scores, promp_scores, coher_scores = scores().three_scores

    test_features_adv = []
    #         test_handmark_normalized_scores = []
    for i in correspond_adv_id_set:
    #             print("DONE HERE2", i)
    #             print(i)
        if (i in features_adv.keys() or (i - 100000) in features_adv.keys()) or i in basic_scores and i in promp_scores and i in coher_scores:
    #                 print("DONE2")
            temp_i = i
            if temp_i > 100000:
                temp_i = temp_i - 100000
                print("bigger", temp_i)
    #         temp_i = str(temp_i)
            temp_features_adv = features_adv[temp_i]
    #         temp_features_adv = np.append(temp_features_adv, basic_scores[i])
    #         temp_features_adv = np.append(temp_features_adv, coher_scores[i])
    #         temp_features_adv = np.append(temp_features_adv, promp_scores[i])
    #                 temp_features.append(basic_scores[i])
    #                 temp_features.append(coher_scores[i])
    #                 temp_features.append(promp_scores[i])
    #                 temp_features.append(1.0)
    #                 temp_features.append(1.0)
            test_features_adv.append(temp_features_adv)

    test_features_adv = np.array(test_features_adv)
    print(test_features_adv)
    pred_scores = xgb_rg.predict(test_features_adv)
    print("pred_scores", pred_scores)
    test_predict_scores = []
    for i in range(len(correspond_adv_id_set)):
    #     min_value = 2
    #     max_value = 12
        overall_score = pred_scores[i] * (max_value - min_value) + min_value
        test_predict_scores.append(overall_score)

    file = open("predictions/prompt5/"+testcases_name+"_upscaled.csv", 'w+', newline ='') 

    # writing the data into the file 
    with file:     
        write = csv.writer(file) 
        write.writerows(map(lambda x: [x], test_predict_scores))
    print("save to csv: ", testcases_name)