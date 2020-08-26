
# PREDICT USING MODEL

import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
from keras.models import model_from_json
from nea.my_layers import MeanOverTime

# load json and create model

#testcases = ['remove_test', 'repeat_test_conc1', 'repeat_test_conc2', 'repeat_test_intro1', 'repeat_test_intro2', 'repeat_test_middle1', 'repeat_test_middle2', 'repeat_test_middle3', 'songs_test_beg', 'songs_test_end', 'speeches_test_beg', 'speeches_test_end', 'end_escalation', 'start_escalation']

#testcases = ['uf_1_beg', 'uf_1_mid', 'uf_1_end', 'ut_1_beg', 'ut_1_mid', 'ut_1_end', 'wiki_1_beg', 'wiki_1_mid', 'wiki_1_end', 'wiki_rel_1_beg', 'wiki_rel_1_mid', 'wiki_rel_1_end', 'wiki_topic_1_beg', 'wiki_topic_1_mid', 'wiki_topic_1_end']
#testcases = ['rc_5_beg', 'rc_5_end', 'rc_5_mid']
#cases = ['bound', 'unbound']
testcases = ['svo_all']
for tc in testcases:
	for prompt in range(1, 9):
		for i in range(0, 5):
		
			print("Word: %s, prompt: %d, Fold: %d" %(tc, prompt, i))
			json_file = open('/mnt/data/rajivratn/nea/nea/output/prompt%d/prompt%d_output_fold%d/model_arch.json' %(prompt, prompt, i), 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			loaded_model = model_from_json(loaded_model_json, custom_objects={'MeanOverTime':MeanOverTime})
			#print("Done")
			# load weights into new model

			loaded_model.load_weights("output/prompt%d/prompt%d_output_fold%d/best_model_weights.h5" %(prompt, prompt, i))
			#print("Loaded model from disk")

			print("Predicting...")
			import numpy as np
			test_X = np.loadtxt('svo_all/test_x_svo_all_prompt%d.txt'%(prompt), dtype=int)


			predictions = loaded_model.predict(test_X)
			#print(predictions)
			print(predictions.shape)
			sav_dir = '/mnt/data/rajivratn/nea/nea/results/svo_all/prompt%d'%(prompt)
			try:
				os.mkdir(sav_dir)
			except OSError:
				print("Failed")
			else:
				print("Created")

			
			np.savetxt('/mnt/data/rajivratn/nea/nea/results/svo_all/prompt%d/results_%s_prompt%d_sent_fold%d.txt' %(prompt, tc, prompt, i), predictions, fmt='%f')
