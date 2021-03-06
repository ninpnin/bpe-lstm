from tokenizers import Tokenizer
import sys
import pickle
import numpy as np
from build_bpe import cleanup
import os

tokenizer = Tokenizer.from_file("bpe-fi.tokenizer.json")

print(tokenizer)
#dfolder = "../../Data/wiki/fi/"
dfolder = "../../Data/finovels/"
files = os.listdir(dfolder)

print("Read files from", dfolder)
print("...")
#s = open(dpath).read().lower()

lines = []

for dpath in files:
    with open(dfolder + dpath) as f:
        print("File:", dpath)

        for line in f:
            clean_line = cleanup(line)
            lines.append(clean_line)

#print("Encode", s[:100], len(s))
print("ENCODE")
encoded_l = tokenizer.encode_batch(lines)

print("Done.")

output = ""
output_is = []
for encoded in encoded_l:
    data = encoded.tokens
    data = " ".join(data)

    output += " " + data

    for ix in encoded.ids:
        output_is.append(ix)

outf = open("out.txt", "w")
outf.write(output)
outf.close()

output_is = np.array(output_is)
np.save("out.npy", output_is)
