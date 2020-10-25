from tokenizers import CharBPETokenizer, BertWordPieceTokenizer, SentencePieceBPETokenizer
from os import listdir
from os.path import isfile, join
import re

def cleanup(s):
    s = s.lower()
    regex = re.compile('[^a-zåäö \.,!?:\n]')
    s = regex.sub('', s)
    return s

def build_bpe(vocab_size=10000):
    # Initialize a tokenizer
    tokenizer = SentencePieceBPETokenizer()

    #mypath = "../../Downloads/riksdagens_protokoll_1920-2020/annual"
    mypath = "../../Desktop/cood/python/machine-learning/old-school/markov-lstm-killer/data/fi"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print("ONL", onlyfiles)

    paths = [mypath + "/" + f for f in onlyfiles]

    #paths = paths[:5]

    # COPY FILES
    txts = []
    for path, fname in zip(paths, onlyfiles):
        if path[-4:] == ".txt":
            localpath = "data/" + fname
            txts.append(localpath)

            infile = open(path)
            outfile = open(localpath, "w")

            for line in infile:
                clean_line = cleanup(line) + "\n"
                outfile.write(clean_line)

            outfile.close()


    # Then train it!
    #tokenizer.train([ "../../Downloads/riksdagens_protokoll_1920-2020/annual/prot_2019.txt" ], vocab_size=15000)
    tokenizer.train(txts, vocab_size=vocab_size)

    # Now, let's use it:
    s = "Det politiska arbetet har redan börjat på olika sätt, med resor, besök, möten, politikutveckling, motionsskrivande och mycket annat. Jag har sett att ni redan har varit aktiva under ett antal veckor, och jag kan försäkra er att det även gäller talmanspresidiet. Nu är det dags att med tillförsikt påbörja ett nytt riksdagsår. Jag hoppas att ni alla ser fram emot det lika myck­et som jag gör."
    #s = "Ite en oo viel mitää hyvää kyl sielt syöny."
    #s = "ja kieltämät siihe tommoste kokonaisii sanoi merkitsevät tavumerkit on huomattavasti näppärämpii ku ääniä tarkottavat aakkoset joist pitää rakentaa jokane sana"
    encoded = tokenizer.encode(s)

    print(encoded.ids)
    print(encoded.tokens)
    # And finally save it somewhere
    tokenizer.save("./bpe-fi.tokenizer.json")

if __name__ == '__main__':
    build_bpe(vocab_size=10000)