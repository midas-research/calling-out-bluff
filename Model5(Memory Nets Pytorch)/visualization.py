from captum.attr import visualization
import numpy as np 
from os.path import isfile, join
from os import listdir
import argparse
import data
from heatmap import generate 
import time
from datetime import datetime

all_vocab = None
def load_glove(args):
	global all_vocab
	print("Loading Glove.....")
	t1 = time.time()
	word_to_index, word_to_vec, index_to_word = data.load_glove(w_vocab=all_vocab, token_num=args.token_num, dim=args.emb_size)
	word_to_vec = np.array(word_to_vec, dtype=np.float32)
	t2 = time.time()
	print(f"Finished loading Glove!, time cost = {(t2-t1):.4f}s\n")
	return word_to_index, word_to_vec, index_to_word


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='visualization')
	parser.add_argument('--set_id', type=int, required=True, help='set id')
	parser.add_argument('--emb_size', type=int, default=300, help="Embedding size for sentences.")
	parser.add_argument('--token_num', type=int, default=42, help="The number of token in glove (6, 42).")
	parser.add_argument('--feature_size', type=int, default=100, help="Feature size.")
	parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to train for.")
	parser.add_argument('--test_freq', type=int, default=20, help="Evaluate and print results every x epochs.")
	parser.add_argument('--hops', type=int, default=3, help="Number of hops in the Memory Network.")
	parser.add_argument('--lr', type=float, default=0.002, help="Learning rate.")
	parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
	parser.add_argument('--l2_lambda', type=float, default=0.3, help="Lambda for l2 loss.")
	parser.add_argument('--num_samples', type=int, default=1, help="Number of samples selected as memories for each score.")
	parser.add_argument('--epsilon', type=float, default=0.1, help="Epsilon value for Adam Optimizer.")
	parser.add_argument('--max_grad_norm', type=float, default=10.0, help="Clip gradients to this norm.")
	parser.add_argument('--keep_prob', type=float, default=0.9, help="Keep probability for dropout.")
	args = parser.parse_args() 

	checkpoints_dir = 'checkpoints/'
	attr_npy_dir = join(checkpoints_dir, str(args.set_id)) 
	_attributes_files_ = listdir(attr_npy_dir)

	test_essay_contents, test_essay_scores, test_essay_ids = data.load_data('test', args.set_id)
	train_essay_contents, train_essay_scores, train_essay_ids = data.load_data('train' ,args.set_id)
	dev_essay_contents, dev_essay_scores, dev_essay_ids = data.load_data('dev', args.set_id)

	train_sent_size_list = list(map(len, [content for content in train_essay_contents]))
	max_sent_size = max(train_sent_size_list)
	all_vocab = data.all_vocab(dev_essay_contents, train_essay_contents, test_essay_contents)

	word_to_index, word_to_vec, index_to_word = load_glove(args)

	test_contents_idx = data.vectorize_data(test_essay_contents, word_to_index, max_sent_size)

	np.random.shuffle(_attributes_files_)
	for attr_file in _attributes_files_[:5]:
		sent = []
		idx = attr_file.split('_')[2]
		idx = int(idx)
		attr = np.load(join(attr_npy_dir, attr_file)) 
		content = test_contents_idx[idx]
		for w_idx in content:
			if w_idx == 0:
				sent.append('[UNK]')
			else:
				sent.append(index_to_word[w_idx])
		# sent = [index_to_word[w_idx] for w_idx in content if w_idx != 0 else '[UNK]']
		attr = np.squeeze(attr)
		extra_len = len(attr) - len(sent)
		extra_sent = ['[UNK]'] * extra_len
		print(extra_sent)
		print(len(extra_sent))
		sent += extra_sent
		print('calling generate')
		generate(sent, attr, str(datetime.now()) + '_' + attr_file.split('.')[0]  + '.tex', rescale_value=True)
		print('finished generate')

		





