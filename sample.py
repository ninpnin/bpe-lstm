import tensorflow as tf
import numpy as np
from tokenizers import Tokenizer
import random
from train import config, model_str

INPUT_LEN = config["input_len"]
DIM = config["dim"]
DIM = config["dim"]
OUTPUT_LEN = 40
model_filename = model_str(config)

model = tf.keras.models.load_model('./saved_models/' + model_filename)

tokenizer = Tokenizer.from_file("bpe-fi.tokenizer.json")
vocab = tokenizer.get_vocab()
inv_vocab = {value: key for key, value in vocab.items()}
indices = list(inv_vocab.keys())

x0 = "moi moi mitä sinulle kullu väinö:"
x0 = np.array([tokenizer.encode(x0).ids])[:,-INPUT_LEN:]
print("X0 shape", x0.shape)

def probs(x, temp=1.0):
	x_prime = x * temp
	x_prime = np.exp(x_prime)
	return x_prime / np.sum(x_prime)

output = []
for i in range(OUTPUT_LEN):
	new_wordpiece = inv_vocab[int(x0[0,-1])]
	output.append(new_wordpiece)
	print(new_wordpiece)

	yi = model(x0)
	print("yi", yi.shape)
	p = probs(yi, temp=1.0)[0]
	x_new = np.array([random.choices(indices, weights=p)])
	x_old = x0[:,1:]
	x0 = tf.concat([x0[:,1:], x_new], axis=1)
	print("x0", x0.shape)

print(" ".join(output))
output = "".join(output)
output = output.replace("▁", " ")
print(output)
