# PREDICT USING MODEL
import sys 
argslist = list(sys.argv)
print(argslist)

import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
from keras.models import model_from_json
from nea.my_layers import MeanOverTime
from nea.my_layers import Conv1DWithMasking
import nea.asap_reader as dataset
import pandas as pd
import numpy as np
import statistics
from keras.preprocessing import sequence

prompt_id = int(argslist[1])
adv_file = argslist[2]
# prompt_id = 1
print("PromptID is: ", prompt_id)
print("Adversarial File: ", adv_file)

fold = 5
vocab_size = 4000
maxlen = 0 #Maximum allowed number of words during training. '0' means no limit

advtestcase_name = adv_file[:-4]
(test_x), vocab, vocab_size, overal_maxlen, num_outputs = dataset.get_data(adv_file, prompt_id, vocab_size, maxlen, tokenize_text=True, to_lower=True, sort_by_len=False)
test_X = sequence.pad_sequences(test_x)

json_file = open('nea/nea/output/prompt%d/prompt%d_output_fold0/model_arch.json' %(prompt_id, prompt_id), 'r')
loaded_model_json = json_file.read()
json_file.close() 
loaded_model_0 = model_from_json(loaded_model_json, custom_objects={'MeanOverTime':MeanOverTime})
loaded_model_0.load_weights("nea/nea/output/prompt%d/prompt%d_output_fold0/best_model_weights.h5" %(prompt_id, prompt_id))
    
json_file = open('nea/nea/output/prompt%d/prompt%d_output_fold1/model_arch.json' %(prompt_id, prompt_id), 'r')
loaded_model_json = json_file.read()
json_file.close() 
loaded_model_1 = model_from_json(loaded_model_json, custom_objects={'MeanOverTime':MeanOverTime})
loaded_model_1.load_weights("nea/nea/output/prompt%d/prompt%d_output_fold1/best_model_weights.h5" %(prompt_id, prompt_id))
    
json_file = open('nea/nea/output/prompt%d/prompt%d_output_fold2/model_arch.json' %(prompt_id, prompt_id), 'r')
loaded_model_json = json_file.read()
json_file.close() 
loaded_model_2 = model_from_json(loaded_model_json, custom_objects={'MeanOverTime':MeanOverTime})
loaded_model_2.load_weights("nea/nea/output/prompt%d/prompt%d_output_fold2/best_model_weights.h5" %(prompt_id, prompt_id))
    
json_file = open('nea/nea/output/prompt%d/prompt%d_output_fold3/model_arch.json' %(prompt_id, prompt_id), 'r')
loaded_model_json = json_file.read()
json_file.close() 
loaded_model_3 = model_from_json(loaded_model_json, custom_objects={'MeanOverTime':MeanOverTime})
loaded_model_3.load_weights("nea/nea/output/prompt%d/prompt%d_output_fold3/best_model_weights.h5" %(prompt_id, prompt_id))
    
json_file = open('nea/nea/output/prompt%d/prompt%d_output_fold4/model_arch.json' %(prompt_id, prompt_id), 'r')
loaded_model_json = json_file.read()
json_file.close() 
loaded_model_4 = model_from_json(loaded_model_json, custom_objects={'MeanOverTime':MeanOverTime})
loaded_model_4.load_weights("nea/nea/output/prompt%d/prompt%d_output_fold4/best_model_weights.h5" %(prompt_id, prompt_id))
    
predictions_0 = loaded_model_0.predict(test_X)
predictions_1 = loaded_model_1.predict(test_X)
predictions_2 = loaded_model_2.predict(test_X)
predictions_3 = loaded_model_3.predict(test_X)
predictions_4 = loaded_model_4.predict(test_X)

print(len(predictions_0), len(predictions_1), len(predictions_2), len(predictions_3), len(predictions_4))

avg = np.mean([predictions_0, predictions_1, predictions_2, predictions_3, predictions_4], axis=0)
n = len(avg)
print(n)
def upscale(output, b, a):
    real_output = (output * (b - a)) + a
    return real_output

if (prompt_id == 1):
    fin = pd.DataFrame(upscale(avg,12,2))
elif (prompt_id == 2):
    fin = pd.DataFrame(upscale(avg,6,1))
elif (prompt_id == 3 or 4):
    fin = pd.DataFrame(upscale(avg,3,0))
elif (prompt_id == 5 or 6):
    fin = pd.DataFrame(upscale(avg,4,0))
elif (prompt_id == 7):
    fin = pd.DataFrame(upscale(avg,30,0))
elif (prompt_id == 8):
    fin = pd.DataFrame(upscale(avg,60,0))
print(fin.head())
fin.to_csv("lstm_mot_%s.csv"%(advtestcase_name),index=None)

