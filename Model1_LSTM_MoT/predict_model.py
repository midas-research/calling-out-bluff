# PREDICT USING MODEL

import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
from keras.models import model_from_json
from nea.my_layers import MeanOverTime

# load json and create model

#testcases = ['remove_test', 'repeat_test_conc1', 'repeat_test_conc2', 'repeat_test_intro1', 'repeat_test_intro2', 'repeat_test_middle1', 'repeat_test_middle2', 'repeat_test_middle3', 'songs_test_beg', 'songs_test_end', 'speeches_test_beg', 'speeches_test_end', 'end_escalation', 'start_escalation']

#testcases = ['uf_1_beg', 'uf_1_mid', 'uf_1_end', 'ut_1_beg', 'ut_1_mid', 'ut_1_end', 'wiki_1_beg', 'wiki_1_mid', 'wiki_1_end', 'wiki_rel_1_beg', 'wiki_rel_1_mid', 'wiki_rel_1_end', 'wiki_topic_1_beg', 'wiki_topic_1_mid', 'wiki_topic_1_end']
testcases = ['rc_5_beg', 'rc_5_end', 'rc_5_mid']
cases = ['bound', 'unbound']
for tc in testcases:
	for cs in cases:
		for i in range(0, 5):
		
			print("Word: %s, case: %s, Fold: %d" %(tc, cs, i))
			json_file = open('/mnt/data/rajivratn/nea/nea/output/prompt5/prompt5_output_fold%d/model_arch.json' %(i), 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			loaded_model = model_from_json(loaded_model_json, custom_objects={'MeanOverTime':MeanOverTime})
			#print("Done")
			# load weights into new model

			loaded_model.load_weights("output/prompt5/prompt5_output_fold%d/best_model_weights.h5" %(i))
			#print("Loaded model from disk")

			print("Predicting...")
			import numpy as np
			test_X = np.loadtxt('prompt5_new/test_x_%s_%s.txt'%(tc, cs), dtype=int)


			predictions = loaded_model.predict(test_X)
			#print(predictions)
			print(predictions.shape)
			sav_dir = '/mnt/data/rajivratn/nea/nea/results/prompt5_new_new/%s_%s'%(tc, cs)
			try:
				os.mkdir(sav_dir)
			except OSError:
				print("Failed")
			else:
				print("Created")
			np.savetxt('/mnt/data/rajivratn/nea/nea/results/prompt5_new_new/%s_%s/results_%s_%s_sent_fold%d.txt' %(tc, cs, tc, cs, i), predictions, fmt='%f')
