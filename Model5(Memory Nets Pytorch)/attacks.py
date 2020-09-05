from data import tokenize
from nltk import sent_tokenizer
import os 
import random 

def get_avg_length(set_id):
	dev_essay_contents, dev_essay_scores, dev_essay_ids = data.load_data('dev' ,set_id)
	lengths = [len(nltk.sent_tokenizer(essay)) for essay in dev_essay_contents]
	average_length = np.mean(lengths)
	return average_length

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
	tok_sent = tok_sent(sentence) 
	n = len(tok_sent)
	mid_point = n // 2
	start_index = mid_point - k // 2
	end_index = mid_point + k // 2
	for i in range(start_index, end_index + 1):
		tok_sent[i] = '[UNK]'

	return ' '.join(tok_sent)



def get_songs(path):
	file = open(path, 'r')
	songs = file.readlines()[1:]
	return songs  



def get_all_songs():
	song_path = '/home/rajivratn/Maiti/why-this-score/calling-out-bluff/Model5(Memory Nets Pytorch)/AES_testcases/songs'
	song_files = os.listdir(song_path) 
	all_songs = [] 
	for song_file in song_files:
		path = os.path.join(song_path, song_file) 
		songs = get_songs(path) 
		all_songs.append(songs) 
	return all_songs 

def insert_songs_beg(text, set_id):
	avg_l = get_avg_length(set_id)
	ar = random.sample(songs, avg_l)
	s = ''.join(ar)
	res = s+'.'+text
	return res

def insert_songs_end(text, set_id):
	avg_l = get_avg_length(set_id)
	ar = random.sample(songs, avg_l)
	s = ''.join(ar)
	res = text+'.'+s

def insert_songs_mid(text, set_id):
	avg_l = get_avg_length(set_id)
	text_data = tokenize.sent_tokenize(text)
	ar = random.sample(songs, avg_l)
	
	for i in ar:
		text_data.insert(int(len(text_data)/2), i)
	return ''.join(text_data)






	


