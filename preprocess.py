import os
import codecs
import pickle

from nltk.tokenize import sent_tokenize, word_tokenize

from arguments import preprocess_args

def skipgram(args, sentence, i):
    iword = sentence[i]
    left = sentence[max(i - args.window, 0): i]
    right = sentence[i + 1: i + 1 + args.window]
    return iword, [args.unk for _ in range(args.window - len(left))] + left + right + [args.unk for _ in range(args.window - len(right))]

def main(args):
    print("Building vocab...")
    step = 0
    wc = {args.unk: 1}

    with open(args.vocab, 'r', encoding='utf-8') as file:
        for line in file:
            step += 1
            if not step % 1000:
                print("Working on line {:,},000 of the vocab...".format(step // 1000), end='\r')
            line = line.strip()
            if not line:
                continue
            sents = sent_tokenize(line)
            for sent in sents:
                words = word_tokenize(sent)
                for word in words:
                    wc[word] = wc.get(word, 0) + 1

    idx2word = [args.unk] + sorted(wc, key=wc.get, reverse=True)[:args.max_vocab - 1]
    word2idx = {idx2word[idx]: idx for idx, _ in enumerate(idx2word)}
    vocab = set([word for word in word2idx])
    if step > 1000: print("")
    print("Done building vocab.")
    print("Vocab size is {}".format(len(idx2word)))

    print("Converting corpus...")
    step = 0
    data = []
    with open(args.corpus, 'r', encoding='utf-8') as file:
        for line in file:
            step += 1
            if not step % 1000:
                print("Working on line {:,},000 of the corpus...".format(step // 1000), end='\r')
            line = line.strip()
            if not line:
                continue
            sentence = []
            for sent in sents:
                words = word_tokenize(sent)
                for word in words:
                    if word in vocab:
                        sentence.append(word)
                    else:
                        sentence.append(args.unk)
            for i in range(len(sentence)):
                iword, owords = skipgram(args, sentence, i)
                data.append((word2idx[iword], [word2idx[oword] for oword in owords]))

    if step > 1000: print("")
    print("Conversion done.")

    print("Saving data...")
    pickle.dump(wc, open(os.path.join(args.data_dir, 'wc.dat'), 'wb'))
    pickle.dump(vocab, open(os.path.join(args.data_dir, 'vocab.dat'), 'wb'))
    pickle.dump(idx2word, open(os.path.join(args.data_dir, 'idx2word.dat'), 'wb'))
    pickle.dump(word2idx, open(os.path.join(args.data_dir, 'word2idx.dat'), 'wb'))
    pickle.dump(data, open(os.path.join(args.data_dir, 'train.dat'), 'wb'))


if __name__ == '__main__':
    args = preprocess_args()
    main(args)
