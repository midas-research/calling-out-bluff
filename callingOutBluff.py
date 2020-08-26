import os
import argparse
import subprocess
#Sample code
#python Callingbluff.py --prompt 2 --orginal_file _ --adv_file speech_2_blah.csv --models 3 2 5

###############################################################################################################################
## Parse arguments#

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", dest="prompt_id", type=str, metavar='<int>', required=True, help="Promp ID for ASAP dataset. For ASAP-AES dataset, prompts lie in range 1-8 and prompts lie in range 1-10 for ASAP-SAS dataset")
parser.add_argument("--original-file", dest="path_to_orginal_file", type=str, metavar='<str>', required=True, help="The path to the original file")
parser.add_argument("-adv-file", "--adv-file", dest="path_to_adversarial_testfile", type=str, metavar='<str>', required=True, help="The path to the adversarial testfile")
parser.add_argument("--models", nargs='+', type=int, required=True, help="specify ModelIDs from 1 to 5")
args = parser.parse_args()
print(args)                    
###############################################################################################################################

for elem in args.models:
    assert elem in {1, 2, 3, 4, 5}
# print("Done")

for model_id in args.models:
    if model_id == 1:
        print("Selected model 1-> LSTM with MoT layer")
        subprocess.run(['/home/rajivratn/anaconda3/envs/att_SAS/bin/python3.6', 'nea/nea/nea_predict_scores_singleCommand.py', args.prompt_id, args.path_to_original_file, args.path_to_adversarial_testfile])
        print("DONE, FILE SAVED")
        print("#############################################################################")

    if model_id == 2:
        print("Selected model 2-> EASE")
        #present in jupyter notebooks
        #code is on skywalker
        #environments could not setup on matterhorn
    
    if model_id == 3:
        subprocess.run(['/home/rajivratn/anaconda3/envs/skipflow/bin/python3.6', 'skipflow/skipflow_predict_scores_singleCommand.py', args.prompt_id, args.path_to_original_file, args.path_to_adversarial_testfile])
        print("DONE, FILE SAVED")
        print("#############################################################################")
        #REMEMBER TO CHANGE DIRECTORY NAMES
    
    if model_id == 4:
        print("Selected model 4-> BERT Two Stage Learning")
        subprocess.run(['/home/rajivratn/anaconda3/envs/meharpy3/bin/python3.6', 'mehar/fupugec-score/bertTwoStage_predict_scores_singleCommand.py', args.prompt_id, args.path_to_original_file, args.path_to_adversarial_testfile])
        print("DONE, FILE SAVED")
        print("#############################################################################")
        
    if model_id == 5:
        print("Selected model 5-> Memory Networks")
        subprocess.run(['/home/rajivratn/anaconda3/envs/meharpy3/bin/python3.6', 'memory_networks/memoryNets_predict_scores_singleCommand.py', args.prompt_id, args.path_to_original_file, args.path_to_adversarial_testfile])
        print("DONE, FILE SAVED")
        print("#############################################################################")