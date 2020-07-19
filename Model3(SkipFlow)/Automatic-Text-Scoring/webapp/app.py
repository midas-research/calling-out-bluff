from flask import Flask, render_template,request
import numpy as np
import keras.models
import re

import sys 
import os
import json

sys.path.append(os.path.abspath("./models"))
from model import * 

from flask_cors import CORS


app = Flask(__name__)
CORS(app)

#global vars for easy reusability
global model, graph, glove_emb, tokenizer, range_min, range_max
#initialize these variables
# model, graph, glove_emb, tokenizer, range_min, range_max = init('4')

initialize_arr = [init('1'),  init('2'), init('3'), init('4'), init('5'), init('6'), init('7'), init('8')]

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/predict/',methods=['POST'])
def predict():

	data = request.get_data().decode("utf-8");
	data = json.loads(data)
	intext = data['box_val']
	essay_type = data['options']

	# return "hello there!"

	model, graph, glove_emb, tokenizer, range_min, range_max = initialize_arr[int(essay_type)-1]

	return predict_helper(intext, model, graph, glove_emb, tokenizer, range_min, range_max)



def predict_helper(intext, model, graph, glove_emb, tokenizer, range_min, range_max):	
	texts=[]
	labels=[]
	sentences=[]


	texts.append(intext)
	line=intext.strip()
	sentences.append(nltk.tokenize.word_tokenize(line))	

	for i in sentences:
		temp1=np.zeros((1, EMBEDDING_DIM))
		for w in i:
			if(w in glove_emb):
				temp1+=glove_emb[w]
		temp1/=len(i)

	sequences=tokenizer.texts_to_sequences(texts)
	word_index=tokenizer.word_index 

	data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

	print("Predicting: .... ")
	y_pred=model.predict([data])

	y_pred_fin =[int(round(a*(range_max-range_min)+range_min)) for a in y_pred.reshape(1).tolist()]
	print("Prediction: " , y_pred_fin[0])
	return str(y_pred_fin[0])	

if __name__ == '__main__':
    app.run()
