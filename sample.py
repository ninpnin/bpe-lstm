import tensorflow as tf
import numpy as np
from tokenizers import Tokenizer
import random

INPUT_LEN = 5

model = tf.keras.models.load_model('./saved_model')
tokenizer = Tokenizer.from_file("bpe-fi.tokenizer.json")
vocab = tokenizer.get_vocab()
inv_vocab = {value: key for key, value in vocab.items()}

indices = list(inv_vocab.keys())


#x0 = np.array([random.choices(indices, k=INPUT_LEN)])
#print(x0)

x0 = "moi mitä sinulle kullu väinö:"
x0 = np.array([tokenizer.encode(x0).ids])[:,-INPUT_LEN:]
print("X0 shape", x0.shape)

def probs(x, temp=2.0):
	x_prime = x * temp
	x_prime = np.exp(x_prime)
	return x_prime / np.sum(x_prime)

output = ""
for i in range(40):
	output += inv_vocab[int(x0[0,-1])]
	print("x0 shape", x0.shape)
	yi = model(x0)
	print("yi shape", yi.shape)
	p = probs(yi, temp=0.5)[0]
	print("p shape", p.shape)
	x_new = np.array([random.choices(indices, weights=p)])
	print("x_new shape", x_new.shape)

	x_old = x0[:,1:]
	print("x_old shape", x_old.shape)
	x0 = tf.concat([x0[:,1:], x_new], axis=1)

output = output.replace("▁", " ")
print(output)
