from keras import backend as K
import numpy as np


def Conf_matrix(y_true,y_pred,min_rating, max_rating):
	print("entered")
	num_ratings=int(max_rating-min_rating+1)
	conf_matrix=[[0 for i in range(num_ratings)]for j in range(num_ratings)]
	for a,b in zip(K.eval(y_true,y_pred)):
		a=int(round(a))
		b=int(round(b))
		conf_matrix[a-min_rating][b-min_rating]+=1

	return conf_matrix


def histogram(ratings,min_rating,max_rating):
	num_ratings=int(max_rating-min_rating+1)
	hist=[0 for x in range(num_ratings)]
	for r in K.eval(ratings):
		r=int(round(r))
		hist[r-min_rating]+=1
	return hist


def QWK(y_true,y_pred,min_rating=None, max_rating=None):
	if min_rating is None:
		min_rating=K.min(y_true)
		# min_rating=K.min(K.min(y_true),K.min(y_pred))
	if max_rating is None:
		max_rating=K.max(y_true)
		# max_rating=K.max(K.max(y_true),K.max(y_pred))

	conf_matrix= Conf_matrix(y_true,y_pred,K.eval(min_rating),K.eval(max_rating))

	hist_a=histogram(y_true,K.eval(min_rating),K.eval(max_rating))
	hist_b=histogram(y_pred,K.eval(min_rating),K.eval(max_rating))

	num=0.0;denom=0.0

	num_ratings=len(conf_matrix)
	num_items=float(len(K.eval(y_true)))

	for i in range(num_ratings):
		for j in range(num_ratings):

			expected_count=(hist_a[i]*hist_b[j]/num_items)

			d=pow(i-j,2.0)/pow(num_ratings-1,2.0)
			num+=d*conf_matrix[i][j]/num_items
			denom+=d*expected_count/num_items

	return K.variable(1.0 - num/denom)

