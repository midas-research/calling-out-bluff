from os.path import isdir, join 
import os 
import data
from model import MANM
from metric import kappa
import argparse
import time
import numpy as np
from torch import optim
import torch
from captum.attr import IntegratedGradients
from torch.autograd import Variable, Function
from termcolor import colored
from heatmap import generate 
from sys import exit 
import pickle 
from attacks import insert_songs_beg
device = torch.device("cpu")


def get_attributes(contents, memory_contents, ig, model, test_score=None, save=False):
	contents = np.array(contents, dtype=np.int64)
	batched_memory_contents = np.array(memory_contents, dtype=np.int64)
	contents_pt = Variable(torch.from_numpy(contents).to(device))
	batched_memory_contents_pt = Variable(torch.from_numpy(batched_memory_contents).to(device))
	pred_score = model.test(contents, batched_memory_contents) 
	print(colored("Prediction : {}".format(pred_score.item()), "red"))
	embeddings = model.word_to_vec(contents_pt)
	memories = model.word_to_vec(batched_memory_contents_pt)
	if test_score is None:
		attributions, _ = ig.attribute((Variable(embeddings), Variable(memories)), target=pred_score, return_convergence_delta=False)
	else:
		attributions, _ = ig.attribute((Variable(embeddings), Variable(memories)), target=test_score, return_convergence_delta=False)
	
	word_wise_attributions = attributions.sum(dim=2)
	return word_wise_attributions.squeeze(0).detach().cpu().numpy(), pred_score.item() 

def get_sentence(vector, index_to_word):
	sentence = [] 
	for token in vector:
		if token == 0:
			sentence.append('[UNK]')
		else:
			sentence.append(index_to_word[token])

	return sentence


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='interactive model')
	# parser.add_argument('--save_dir', type=str, required=True, help='directory where the results are stored')
	parser.add_argument('--set_id', type=int, required=True, help='set id')
	parser.add_argument('--rw', type=str, default=None, help='random word being used for padding')
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
	parser.add_argument('--mem', action='store_true', help='To Toggle Memory Embeddings')
	parser.add_argument('--epsilon', type=float, default=0.1, help="Epsilon value for Adam Optimizer.")
	parser.add_argument('--max_grad_norm', type=float, default=10.0, help="Clip gradients to this norm.")
	parser.add_argument('--keep_prob', type=float, default=0.9, help="Keep probability for dropout.")
	parser.add_argument('--sample_id', type=int, required=True, help="Sample id for the respective prompt")
	parser.add_argument('--run_name', type=str, required=True, help="Name of the run (for easy tracking of results)")
	parser.add_argument('--save_dir', type=str, default=join('checkpoints', 'main_model'), help='Name of the checkpoints directory')
	parser.add_argument('--song_beg', action='store_true', help='Toggle adding song in beginning')
	args = parser.parse_args()
	print(args)


	train_essay_contents, train_essay_scores, train_essay_ids = data.load_data('train' , args.set_id)
	dev_essay_contents, dev_essay_scores, dev_essay_ids = data.load_data('dev', args.set_id)
	test_essay_contents, test_essay_scores, test_essay_ids = data.load_data('test', args.set_id)
	all_vocab = data.all_vocab(train_essay_contents, dev_essay_contents, test_essay_contents)
	print(f"all_vocab len:{len(all_vocab)}")
	train_sent_size_list = list(map(len, [content for content in train_essay_contents]))
	max_sent_size = max(train_sent_size_list)
	mean_sent_size = int(np.mean(train_sent_size_list))
	min_score = min(train_essay_scores)
	max_score = max(train_essay_scores)

	print('max_score={} \t min_score={}'.format(max_score, min_score))

	print('max train sentence size={} \t mean train sentence size={}\n'.format(max_sent_size, mean_sent_size))
	print("Loading Glove.....")
	t1 = time.time()
	word_to_index, word_to_vec, index_to_word = data.load_glove(w_vocab=all_vocab, token_num=args.token_num, dim=args.emb_size)
	print(len(word_to_index), len(word_to_vec), len(index_to_word))
	# exit() 
	word_to_vec = np.array(word_to_vec, dtype=np.float32)
	t2 = time.time()
	print(f"Finished loading Glove!, time cost = {(t2-t1):.4f}s\n")

	padding_idx = 0
	if args.rw is not None:
		if args.rw not in word_to_index:
			print('Word not in distribution!')
			print('Do you want to continue? The word will be replaced by 0 (y/n)?')
			ans = input()
			if ans.lower() == 'n':
				exit()  
		else:
			padding_idx = word_to_index[args.rw]



	# [train_essay_size, max_sent_size]  type: list
	train_contents_idx = data.vectorize_data(train_essay_contents, word_to_index, max_sent_size)
	# # [dev_essay_size, max_sent_size]  type: list
	dev_contents_idx = data.vectorize_data(dev_essay_contents, word_to_index, max_sent_size)
	# # [test_essay_size, max_sent_size]  type: list
	test_contents_idx = data.vectorize_data(test_essay_contents, word_to_index, max_sent_size)
	score_range = list(range(min_score, max_score + 1))
	memory_contents = []
	memory_scores = []
	for i in score_range:
		for j in range(args.num_samples):
			if i in train_essay_scores:
				score_idx = train_essay_scores.index(i)
				score = train_essay_scores.pop(score_idx)  # score=i
				content = train_contents_idx.pop(score_idx)
				memory_contents.append(content)
				memory_scores.append(score)
			else:
				print(f"score {i} is not in train data")

	memory_size = len(memory_contents)  # actual score_range
	train_scores_index = list(map(lambda x: score_range.index(x), train_essay_scores))


	model = MANM(word_to_vec=word_to_vec, max_sent_size=max_sent_size, memory_num=memory_size, embedding_size=args.emb_size,
				 feature_size=args.feature_size, score_range=len(score_range), hops=args.hops,
				 l2_lambda=args.l2_lambda, keep_prob=args.keep_prob, device=device, mem_embedding=args.mem).to(device)
	model = torch.load(f"{args.save_dir}/save_one_{args.set_id}.ckpt")
	id_to_score = dict(zip(dev_essay_ids, dev_essay_scores))
	id_to_content = dict(zip(dev_essay_ids, dev_essay_contents))

	natural_input = ' '.join(id_to_content[args.sample_id])
	model_input = None
	attack_name = ''
	if args.song_beg:
		perturbed_input = insert_songs_beg(natural_input, args.set_id)
		perturbed_input = data.tokenize(data.clean_str(perturbed_input))
		model_input = data.vectorize_data([perturbed_input], word_to_index, max_sent_size, padding_random=padding_idx) #id_to_content[args.sample_id]
		attack_name = 'add_song_beg'
		print(perturbed_input)
	else:
		natural_input = data.tokenize(data.clean_str(natural_input))
		model_input = data.vectorize_data([natural_input], word_to_index, max_sent_size, padding_random=padding_idx)
		attack_name = 'None'

	ig = IntegratedGradients(model.forward_ig)
	test_score = torch.from_numpy(np.array([id_to_score[args.sample_id] - min_score]))
	print(test_score)
	test_score = torch.as_tensor(np.array([test_score]))
	word_wise_attributions, pred_score = get_attributes(model_input, [memory_contents], ig, model, test_score=test_score) 
	word_wise_attributions = np.squeeze(word_wise_attributions)

	if args.rw is None:
		args.rw = '[UNK]'
	if not isdir(join('checkpoints', args.run_name)):
		os.mkdir(join('checkpoints', args.run_name))
	sentence = get_sentence(model_input[0], index_to_word)

	
	generate(sentence, word_wise_attributions, join('checkpoints', args.run_name, f'set_{args.set_id}_sample_id_{args.sample_id}_{args.rw}_gt_{test_score.item() + min_score}_pred_{pred_score}_attack_name_{attack_name}'+'.tex'), rescale_value=True)











