from data import tokenize
from nltk.tokenize import sent_tokenize
import os 
import random 
import data
import numpy as np 
import math
random.seed(10)

def get_avg_length(set_id):
	dev_essay_contents= data.load_prompts('dev' ,set_id)
	lengths = [len(sent_tokenize(essay)) for essay in dev_essay_contents]
	average_length = np.mean(lengths)
	print('Average Length of the prompts are ', average_length)
	return average_length / 4

def add_p_percent_unk(sentence: str, p: float) -> str:
	tok_sent = tokenize(sentence)
	n = len(tok_sent)
	num_unks = int(p * n)
	mid_point = n // 2
	start_index = mid_point - num_unks // 2
	end_index = mid_point + num_unks // 2
	for i in range(start_index, end_index + 1): 
		tok_sent[i] = '[UNK]'

	return ' '.join(tok_sent)


def add_k_abs_unk(sentence: str, k: int) -> str:
	tok_sent = sent_tokenize(sentence) 
	n = len(tok_sent)
	mid_point = n // 2
	start_index = mid_point - k // 2
	end_index = mid_point + k // 2
	for i in range(start_index, end_index + 1):
		tok_sent[i] = '[UNK]'

	return ' '.join(tok_sent)



def get_songs(path):
	file = open(path, 'r', errors='ignore')
	songs = file.readlines()[1:]
	# print(len(songs))
	return songs  



def get_all_songs():
	song_path = '/home/rajivratn/Maiti/why-this-score/calling-out-bluff/Model5(Memory Nets Pytorch)/AES_testcases/songs'
	song_files = os.listdir(song_path) 
	all_songs = [] 
	for song_file in song_files:
		path = os.path.join(song_path, song_file) 
		songs = get_songs(path) 
		all_songs.extend(songs) 
	return all_songs 

def insert_songs_beg(text, set_id):
	songs = get_all_songs()
	# print(len(songs))
	# print(songs)
	avg_l = get_avg_length(set_id)
	avg_l = int(math.ceil(avg_l))
	ar = random.sample(songs, 1)
	s = ''.join(ar)
	print('song_length ', len(s.split()), 'text length ', len(text.split()))
	res = s+'.'+text
	return res

def insert_songs_end(text, set_id):
	avg_l = get_avg_length(set_id)
	songs = get_all_songs()
	ar = random.sample(songs, avg_l)
	s = ''.join(ar)
	res = text+'.'+s

def insert_songs_mid(text, set_id):
	avg_l = get_avg_length(set_id)
	songs = get_all_songs()
	text_data = tokenize.sent_tokenize(text)
	ar = random.sample(songs, avg_l)
	
	for i in ar:
		text_data.insert(int(len(text_data)/2), i)
	return ''.join(text_data)






	


