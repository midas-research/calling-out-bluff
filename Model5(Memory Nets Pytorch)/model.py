# @title
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable, Function


class MANM(nn.Module):
	def __init__(
		self,
		word_to_vec,
		max_sent_size,
		memory_num,
		embedding_size,
		feature_size,
		score_range,
		hops,
		l2_lambda,
		keep_prob,
		device,
		mem_embedding,
	):
		super(MANM, self).__init__()
		self.cond = False
		self.max_sent_size = max_sent_size
		self.memory_num = memory_num
		self.hops = hops
		self.l2_lambda = l2_lambda
		self.keep_prob = keep_prob
		self.score_range = score_range
		self.feature_size = feature_size
		self.embedding_size = embedding_size
		self.device = device
		self.mem_embedding = mem_embedding
		if self.mem_embedding:
			self.word_to_vec = {
				"Memory": torch.nn.Embedding.from_pretrained(
					torch.from_numpy(word_to_vec), freeze=True
				),
				"Input": torch.nn.Embedding.from_pretrained(
					torch.from_numpy(word_to_vec), freeze=True
				),
			}
		else:
			self.word_to_vec = torch.nn.Embedding.from_pretrained(
				torch.from_numpy(word_to_vec), freeze=True
			)
		print('Word to vec size', len(self.word_to_vec.weight))

		# [embedding_size, max_sent_size]
		self.pos_encoding = (
			self.position_encoding(self.max_sent_size, self.embedding_size)
			.requires_grad_(False)
			.to(self.device)
		)

		# shape [k, d]
		self.A = torch.nn.Embedding(self.feature_size, self.embedding_size).to(
			self.device
		)
		self.B = torch.nn.Embedding(self.feature_size, self.embedding_size).to(
			self.device
		)
		self.C = torch.nn.Embedding(self.feature_size, self.embedding_size).to(
			self.device
		)
		torch.nn.init.xavier_uniform_(self.A.weight)
		torch.nn.init.xavier_uniform_(self.B.weight)
		torch.nn.init.xavier_uniform_(self.C.weight)
		# shape [k, k]
		Rlist = []
		for i in range(self.hops):
			R = torch.nn.Embedding(self.feature_size, self.feature_size).to(self.device)
			torch.nn.init.xavier_uniform_(R.weight)
			Rlist.append(R)
		self.R_list = torch.nn.ModuleList(Rlist)
		# shape [k, r]
		self.W = torch.nn.Embedding(self.feature_size, self.score_range).to(self.device)
		torch.nn.init.xavier_uniform_(self.W.weight)
		# bias in last layer
		self.b = torch.nn.Parameter(torch.randn([self.score_range]))
		self.hooks = False

	def forward(
		self, contents_idx: np.ndarray, memories_idx: np.ndarray, scores: np.ndarray
	):
		contents_idx = (
			torch.from_numpy(contents_idx).to(self.device).requires_grad_(False)
		)
		memories_idx = (
			torch.from_numpy(memories_idx).to(self.device).requires_grad_(False)
		)
		if self.mem_embedding:
			# [batch_size, max_sent_size, embedding_size]
			contents = self.word_to_vec["Input"](contents_idx)
			# [batch_size, memory_num, max_sent_size, embedding_size]
			memories = self.word_to_vec["Memory"](memories_idx)
		else:
			# [batch_size, max_sent_size, embedding_size]
			contents = self.word_to_vec(contents_idx)
			# [batch_size, memory_num, max_sent_size, embedding_size]
			memories = self.word_to_vec(memories_idx)
		# emb_contents [batch_size, d]    d=embedding_size
		# emb_memories [batch_size, memory_num, d]
		emb_contents, emb_memories = self.input_representation_layer(contents, memories)
		dropout = torch.nn.Dropout(p=1 - self.keep_prob)
		if self.cond:
			pass
		else:
			emb_contents = dropout(emb_contents).requires_grad_(False)
		# [batch_size, k] = [batch_size, d] x [d, k]
		u = torch.matmul(emb_contents, self.A.weight.transpose(0, 1))

		for i in range(self.hops):
			prob_vectors, used_emb_memories = self.memory_addressing_layer(
				u, emb_memories
			)  # [batch_size, memory_num]
			u = self.memory_reading_layer(
				i, u, prob_vectors, used_emb_memories
			)  # [batch_size, k]

		# [batch_size, memory_num]   distribution is softmax(logits)
		logits, distribution = self.output_layer(u)
		losser = torch.nn.CrossEntropyLoss()
		scores = torch.from_numpy(scores).requires_grad_(False).to(self.device)
		loss1 = torch.sum(losser(logits, scores))  # score: [batch_size]
		loss2 = (
			torch.sum(self.A.weight ** 2)
			+ torch.sum(self.B.weight ** 2)
			+ torch.sum(self.C.weight ** 2)
			+ torch.sum(self.W.weight ** 2)
			+ torch.sum(self.b ** 2)
		) / 2
		for m in range(self.hops):
			loss2 += torch.sum(self.R_list[m].weight ** 2) / 2
		loss = loss1 + loss2 * self.l2_lambda
		return loss

	def position_encoding(self, sentence_size, embedding_size):
		encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
		ls = sentence_size + 1
		le = embedding_size + 1
		for i in range(1, le):
			for j in range(1, ls):
				encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
		encoding = 1 + 4 * encoding / embedding_size / sentence_size
		pos_encoding = torch.from_numpy(encoding)
		return pos_encoding.transpose(0, 1)

	def input_representation_layer(
		self, contents: torch.Tensor, memories: torch.Tensor
	):
		"""bow"""
		# contents [batch_size, max_sent_size, embedding_size]
		# memories [batch_size, memory_num, max_sent_size, embedding_size]
		# self.pos_encoding: [max_sent_size, embedding_size]
		# emb_contents = torch.sum(contents * self.pos_encoding, dim=1).requires_grad_(
		# 	self.cond
		# )  # *表示对位相乘
		# emb_contents = Variable(torch.sum(contents * self.pos_encoding, dim=1), requires_grad=self.cond)
		# print(memories.shape,  self.pos_encoding.shape)
		emb_contents = torch.sum(contents * self.pos_encoding, dim=1)

		if self.mem_embedding:
			emb_memories = torch.sum(
				memories * self.pos_encoding, dim=2
			).requires_grad_(
				False
			)  # *表示对位相乘

		# emb_memories = torch.sum(memories * self.pos_encoding, dim=2).requires_grad_(
		# 	self.cond
		# )  # *表示对位相乘
		emb_memories = torch.sum(memories * self.pos_encoding, dim=2)
		return emb_contents, emb_memories

	def memory_addressing_layer(self, u, emb_memories):
		dropout = torch.nn.Dropout(p=1 - self.keep_prob)
		if self.cond:
			used_emb_memories = emb_memories
		else:
			used_emb_memories = dropout(emb_memories)#.requires_grad_(False)
		# [batch_size, memory_num, k] = [batch_size, memory_num, d] x [d, k]
		trans_emb_memories = torch.matmul(
			used_emb_memories, self.B.weight.transpose(0, 1)
		)
		# dot product
		# [batch_size, memory_num, k] <- [batch_size, k]
		trans_emb_contents = u.unsqueeze(dim=1)
		# product [batch_size, memory_num]  *表示对位相乘
		product = torch.sum(
			trans_emb_contents * trans_emb_memories, dim=-1
		)  # 对最后一维进行sum
		# prob_vectors [batch_size, memory_num]
		prob_vectors = F.softmax(product, dim=-1)
		return prob_vectors, used_emb_memories

	def memory_reading_layer(self, i, u, prob_vectors, used_emb_memories):
		# [batch_size, memory_num, 1]
		prob_vectors = torch.unsqueeze(prob_vectors, dim=2)
		# [batch_size * memory_num, d]
		memo_temp = used_emb_memories.view(-1, self.embedding_size)
		# print("used_emb_memories size: ", memo_temp.shape)
		# [d, batch_size * memory_num]
		memo_temp = memo_temp.transpose(0, 1)
		# [k, batch_size * memory_num]
		product = torch.matmul(self.C.weight, memo_temp)
		# print(product.shape)
		# [batch_size, memory_num, k]
		product = torch.reshape(
			product.transpose(0, 1), [-1, self.memory_num, self.feature_size]
		)
		# product = torch.matmul(used_emb_memories, self.C.weight.transpose(0,1))
		# [batch_size, k]
		o = torch.sum(prob_vectors * product, dim=1)
		# [batch_size, k]
		u = F.relu(torch.matmul((o + u), self.R_list[i].weight))
		return u

	def output_layer(self, u):
		# [batch_size, score_range]
		logits = torch.matmul(u, self.W.weight) + self.b
		distribution = F.softmax(logits, dim=1)
		return logits, distribution

	def test(self, contents_idx, memories_idx):
		contents_idx = (
			torch.from_numpy(contents_idx).to(self.device).requires_grad_(False)
		)
		memories_idx = (
			torch.from_numpy(memories_idx).to(self.device).requires_grad_(False)
		)
		if self.mem_embedding:
			# [batch_size, max_sent_size, embedding_size]
			contents = self.word_to_vec["Input"](contents_idx)
			# [batch_size, memory_num, max_sent_size, embedding_size]
			memories = self.word_to_vec["Memory"](memories_idx)
		else:
			# [batch_size, max_sent_size, embedding_size]
			contents = self.word_to_vec(contents_idx)
			# [batch_size, memory_num, max_sent_size, embedding_size]
			memories = self.word_to_vec(memories_idx)
		# emb_contents [batch_size, d]    d=embedding_size
		# emb_memories [batch_size, memory_num, d]
		emb_contents, emb_memories = self.input_representation_layer(contents, memories)
		self.keep_prob = 1
		# [batch_size, k] = [batch_size, d] x [d, k]
		u = torch.matmul(emb_contents, self.A.weight.transpose(0, 1))

		for i in range(self.hops):
			prob_vectors, used_emb_memories = self.memory_addressing_layer(
				u, emb_memories
			)  # [batch_size, memory_num]
			u = self.memory_reading_layer(
				i, u, prob_vectors, used_emb_memories
			)  # [batch_size, k]

		# [batch_size, memory_num]
		logits, distribution = self.output_layer(u)
		print(distribution.shape, 'distribution.shape')
		# print(distribution)
		# [batch_size]
		pred_scores = torch.argmax(distribution, dim=1)
		return pred_scores


	def forward_ig(self, contents, memories):
		emb_contents, emb_memories = self.input_representation_layer(contents, memories)
		dropout = torch.nn.Dropout(p=1 - self.keep_prob)
		if self.cond:
			pass
		else:
			emb_contents = dropout(emb_contents)#.requires_grad_(False)
		# [batch_size, k] = [batch_size, d] x [d, k]
		u = torch.matmul(emb_contents, self.A.weight.transpose(0, 1))

		for i in range(self.hops):
			prob_vectors, used_emb_memories = self.memory_addressing_layer(
				u, emb_memories
			)  # [batch_size, memory_num]
			u = self.memory_reading_layer(
				i, u, prob_vectors, used_emb_memories
			)  # [batch_size, k]

		# [batch_size, memory_num]   distribution is softmax(logits)
		logits, distribution = self.output_layer(u)
		return distribution



	def save_weights(self, path):
		with open(path, "wb+") as f:
			torch.save(self.state_dict(), f)

	def load_weights(self, path):
		with open(path, "rb") as f:
			self.load_state_dict(torch.load(f))
