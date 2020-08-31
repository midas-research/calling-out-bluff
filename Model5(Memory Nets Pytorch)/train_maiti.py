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
import os 
from sys import exit 
import pickle 
device = torch.device("cpu")


def seed_everything():
	np.random.seed(42)

def get_random_word(word_to_index):
	padding_wrd = np.random.choice(list(word_to_index.keys()))
	padding_num = word_to_index[padding_wrd]
	return padding_num, padding_wrd


def get_attributes(contents, memory_contents, ig, model, test_score=None, save=False):
	contents = np.array(contents, dtype=np.int64)
	batched_memory_contents = np.array(memory_contents, dtype=np.int64)
	contents_pt = Variable(torch.from_numpy(contents).to(device))
	batched_memory_contents_pt = Variable(torch.from_numpy(batched_memory_contents).to(device))
	pred_score = model.test(contents, batched_memory_contents) 
	embeddings = model.word_to_vec(contents_pt)
	memories = model.word_to_vec(batched_memory_contents_pt)
	if test_score is None:
		attributions, _ = ig.attribute((Variable(embeddings), Variable(memories)), target=pred_score, return_convergence_delta=False)
	else:
		attributions, _ = ig.attribute((Variable(embeddings), Variable(memories)), target=test_score, return_convergence_delta=False)
	
	word_wise_attributions = attributions.sum(dim=2)
	return word_wise_attributions.squeeze(0).detach().cpu().numpy()

def main(args, set_id):
	bkp = 0
	if torch.cuda.is_available():
		print(f"Using GPU:{args.gpu_id}")
		device = torch.device("cuda")
		torch.cuda.set_device(args.gpu_id)
	else:
		print("!!! Using CPU")
		device = torch.device("cpu")

	timestamp = time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())
	out_file = "./logs/set{}_{}.txt".format(set_id, timestamp)
	with open(out_file, 'w', encoding='utf-8') as f:
		for key, value in args.__dict__.items():
			f.write("{}={}".format(key, value))
			f.write("\n")

	# read training, dev and test data
	train_essay_contents, train_essay_scores, train_essay_ids = data.load_data('train' ,set_id)
	dev_essay_contents, dev_essay_scores, dev_essay_ids = data.load_data('dev', set_id)
	test_essay_contents, test_essay_scores, test_essay_ids = data.load_data('test', set_id)
	min_score = min(train_essay_scores)
	max_score = max(train_essay_scores)
	if set_id == 7:
		min_score, max_score = 0, 30
	elif set_id == 8:
		min_score, max_score = 0, 60
	score_range = list(range(min_score, max_score + 1))
	# get the vocabulary of training, dev and test datasets.
	all_vocab = data.all_vocab(train_essay_contents, dev_essay_contents, test_essay_contents)
	print(f"all_vocab len:{len(all_vocab)}")

	# get the length of longest essay in training set
	train_sent_size_list = list(map(len, [content for content in train_essay_contents]))
	max_sent_size = max(train_sent_size_list)
	mean_sent_size = int(np.mean(train_sent_size_list))
	print('max_score={} \t min_score={}'.format(max_score, min_score))
	print('max train sentence size={} \t mean train sentence size={}\n'.format(max_sent_size, mean_sent_size))
	with open(out_file, 'a', encoding='utf-8') as f:
		f.write('\n')
		f.write('max_score={} \t min_score={}\n'.format(max_score, min_score))
		f.write('max sentence size={} \t mean sentence size={}\n'.format(max_sent_size, mean_sent_size))

	# loading glove. Only select words which appear in vocabulary.
	print("Loading Glove.....")
	t1 = time.time()
	word_to_index, word_to_vec, index_to_word = data.load_glove(w_vocab=all_vocab, token_num=args.token_num, dim=args.emb_size)
	print(len(word_to_index), len(word_to_vec), len(index_to_word))
	word_to_vec = np.array(word_to_vec, dtype=np.float32)
	t2 = time.time()
	print(f"Finished loading Glove!, time cost = {(t2-t1):.4f}s\n")

	# [train_essay_size, max_sent_size]  type: list
	train_contents_idx = data.vectorize_data(train_essay_contents, word_to_index, max_sent_size)
	# [dev_essay_size, max_sent_size]  type: list
	dev_contents_idx = data.vectorize_data(dev_essay_contents, word_to_index, max_sent_size)
	# [test_essay_size, max_sent_size]  type: list
	test_contents_idx = data.vectorize_data(test_essay_contents, word_to_index, max_sent_size)

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

	# data size
	n_train = len(train_contents_idx)
	n_dev = len(dev_contents_idx)
	n_test = len(test_contents_idx)

	start_list = list(range(0, n_train - args.batch_size, args.batch_size))
	end_list = list(range(args.batch_size, n_train, args.batch_size))
	batches = zip(start_list, end_list)
	batches = [(start, end) for start, end in batches]
	if end_list[len(end_list)-1] != n_train-1:
		batches.append((end_list[len(end_list)-1], n_train-1))

	# model
	model = MANM(word_to_vec=word_to_vec, max_sent_size=max_sent_size, memory_num=memory_size, embedding_size=args.emb_size,
				 feature_size=args.feature_size, score_range=len(score_range), hops=args.hops,
				 l2_lambda=args.l2_lambda, keep_prob=args.keep_prob, device=device, mem_embedding=args.mem).to(device)

	# print('Shape of word to vec weight', model.word_to_vec.size())

	optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

	print("----------begin training----------")
	t1 = time.time()
	dev_kappa_result = 0.0
	for ep in range(1, args.epochs+1):
		t2 = time.time()
		total_loss = 0
		np.random.shuffle(batches)
		for start, end in batches:
			contents = np.array(train_contents_idx[start:end], dtype=np.int64)
			scores_index = np.array(train_scores_index[start:end], dtype=np.int64)
			batched_memory_contents = np.array([memory_contents]*(end-start), dtype=np.int64)
			optimizer.zero_grad()
			loss = model(contents, batched_memory_contents, scores_index)
			total_loss += loss.item()
			loss.backward()
			optimizer.step()
		t3 = time.time()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
		scheduler.step(ep)
		print(f"epoch {ep}/{args.epochs}: all loss={total_loss:.3f}, "
			  f"loss/triple={(total_loss/train_essay_contents.__len__()):.6f}, " f"time cost={(t3-t2):.4f}")
		with open(out_file, 'a', encoding='utf-8') as f:
			f.write("epoch {}: total_loss={:.3f}, loss/triple={:.6f}\n".format(ep, total_loss, total_loss/train_essay_contents.__len__()))
		# begin evaluation
		if ep % args.test_freq == 0 or ep == args.epochs:
			print("------------------------------------")
			mid1 = round(n_dev/3)
			mid2 = round(n_dev/3)*2
			dev_batches = [(0, mid1), (mid1, mid2), (mid2, n_dev)]
			all_pred_scores = []
			for start, end in dev_batches:
				dev_contents = np.array(dev_contents_idx[start:end], dtype=np.int64)
				batched_memory_contents = np.array([memory_contents]*dev_contents.shape[0], dtype=np.int64)
				pred_scores = model.test(dev_contents, batched_memory_contents).cpu().numpy()
				pred_scores = np.add(pred_scores, min_score)
				all_pred_scores += list(pred_scores)
			dev_kappa_result = kappa(dev_essay_scores, all_pred_scores, weights='quadratic')
			print(f"kappa result={dev_kappa_result}")
			print("------------------------------------")
			with open(out_file, 'a', encoding='utf-8') as f:
				f.write("------------------------------------\n")
				f.write("kappa result={}\n".format(dev_kappa_result))
				f.write("------------------------------------\n")
			if dev_kappa_result>bkp:
				bkp=dev_kappa_result
				# model.save_weights(f"../checkpoints/save_one_{args.set_id}.ckpt")
				if not os.path.isdir(f'{args.save_dir}'):
					os.mkdir(f'{args.save_dir}')
				# torch.save(model.state_dict(), f"{args.save_dir}/save_one_{set_id}.ckpt")
				torch.save(model, f"{args.save_dir}/save_one_{set_id}.ckpt")
	# torch.cuda.empty_cache()

	# model.load_state_dict(torch.load(f"{args.save_dir}/save_one_{set_id}.ckpt"))
	# ig = IntegratedGradients(model.forward_ig)
	# for start in range(n_test):
	# 	word_wise_attributions = get_attributes(test_contents_idx[start:start+1], [memory_contents], ig, model)
	# 	save_str = f"{set_id}/attributions_{set_id}_{start}_testset_on_predicted"
		
	# 	if not os.path.isdir(f'{args.save_dir}/{set_id}'):
	# 		os.mkdir(f'{args.save_dir}/{set_id}')

	# 	np.save(f"{args.save_dir}/{save_str}", word_wise_attributions)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='MANN')
	parser.add_argument('--gpu_id', type=int, default=0)
	# parser.add_argument('--set_id', type=int, default=1, help="essay set id, 1 <= id <= 8.")
	parser.add_argument('--emb_size', type=int, default=300, help="Embedding size for sentences.")
	parser.add_argument('--token_num', type=int, default=42, help="The number of token in glove (6, 42).")
	parser.add_argument('--feature_size', type=int, default=100, help="Feature size.")
	parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to train for.")
	parser.add_argument('--test_freq', type=int, default=20, help="Evaluate and print results every x epochs.")
	parser.add_argument('--hops', type=int, default=3, help="Number of hops in the Memory Network.")
	parser.add_argument('--lr', type=float, default=0.002, help="Learning rate.") #make learning rate 0.0001
	parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
	parser.add_argument('--l2_lambda', type=float, default=0.3, help="Lambda for l2 loss.")
	parser.add_argument('--num_samples', type=int, default=1, help="Number of samples selected as memories for each score.")
	parser.add_argument('--epsilon', type=float, default=0.1, help="Epsilon value for Adam Optimizer.")
	parser.add_argument('--max_grad_norm', type=float, default=10.0, help="Clip gradients to this norm.")
	parser.add_argument('--keep_prob', type=float, default=0.9, help="Keep probability for dropout.")
	parser.add_argument('--mem', action='store_true', help='To Toggle Memory Embeddings')
	parser.add_argument('--nzp', action='store_true', help='To use non-zero padding')
	parser.add_argument('--save_dir', required=True, help='checkpoints directory to save the result')
	parser.add_argument('--output_freq', default=10, help='Frequency at which to pickle the test set')
	args = parser.parse_args()
	# bkp=0
	seed_everything() 
	print(args)
	main(args, 1)
	main(args, 3)

	