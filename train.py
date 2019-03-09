import os
import pickle
import random
import argparse
import torch as t
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from model import SkipGram_NS
from arguments import train_args

from tensorboardX import SummaryWriter

class Word2VecDataset(Dataset):

    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def train(args):
    device = torch.device('cuda' if args.cuda else 'cpu')

    writer = SummaryWriter(log_dir=args.log_dir)

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    idx2word = pickle.load(open(os.path.join(args.data_dir, 'idx2word.dat'), 'rb'))

    if args.weights:
        wc = pickle.load(open(os.path.join(args.data_dir, 'wc.dat'), 'rb'))
        word_frequency = np.array([wc[word] for word in idx2word])
        word_frequency = word_frequency / word_frequency.sum()
        ws = 1 - np.sqrt(args.ss_t / word_frequency)
        ws = np.clip(ws, 0, 1)
        word_frequency = np.power(word_frequency, 0.75)
        word_frequency = word_frequency / word_frequency.sum()
        weights = torch.FloatTensor(word_frequency)
    else:
        weights = None    


    vocab_size = len(idx2word)
    print("Vocab size is {:,}".format(vocab_size))

    model = SkipGram_NS(vocab_size=vocab_size, embedding_size=args.e_dim, n_negs=args.n_negs)
    model = model.to(device)
    optim = Adam(model.parameters())

    modelpath = os.path.join(args.save_dir, '{}.pt'.format(args.name))
    if os.path.isfile(modelpath) and args.resume:
        model.load_state_dict(t.load(modelpath))

    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))
    if os.path.isfile(optimpath) and args.resume:
        optim.load_state_dict(t.load(optimpath))

    dataset = Word2VecDataset(os.path.join(args.data_dir, 'train.dat'))
    dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
    total_batches = int(np.ceil(len(dataset) / args.mb))

    for epoch in range(1, args.epoch + 1):
        for batch_idx, (iword, owords) in enumerate(dataloader):

            context_size = owords.size()[1]
            if weights is not None:
                nwords = torch.multinomial(weights, args.mb * context_size * args.n_negs, replacement=True).view(args.mb, -1)
            else:
                nwords = torch.FloatTensor(args.mb, context_size * args.n_negs).uniform_(0, vocab_size - 1).long()

            iword, owords, nwords = iword.to(device), owords.long().to(device), nwords.long().to(device)
            loss = model(iword, owords, nwords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print("[Epoch {}/{}]\tIter: {}/{} ({:.0%})\tLoss: {:8.5f}".format(
            	epoch, args.epoch, batch_idx + 1, total_batches, (batch_idx + 1) / total_batches, loss.item()), end='\r')

        writer.add_scalar('train_loss', loss.item(), epoch)
        print("")    

    print("Saving Embedding Matrix")
    idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(args.data_dir, 'idx2vec.dat'), 'wb'))

    print("Saving Model...")
    t.save(model.state_dict(), os.path.join(args.save_dir, '{}.pt'.format(args.name)))
    t.save(optim.state_dict(), os.path.join(args.save_dir, '{}.optim.pt'.format(args.name)))

    print("Done")


if __name__ == '__main__':
    train(train_args())
