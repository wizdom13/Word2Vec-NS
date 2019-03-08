
import numpy as np
import torch
import torch.nn as nn

class SkipGram_NS(nn.Module):

    def __init__(self, vocab_size=20000, embedding_size=300, padding_idx=0, n_negs=20):
        super(SkipGram_NS, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_negs = n_negs
        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)

        initrange = 0.5 / self.embedding_size
        nn.init.uniform_(self.ivectors.weight.data, -initrange, initrange)
        nn.init.constant_(self.ovectors.weight.data, 0)

    def forward(self, iword, owords, nwords):
        context_size = owords.size()[1]
        ivectors = self.ivectors(iword).unsqueeze(2)
        ovectors = self.ovectors(owords)
        nvectors = self.ovectors(nwords).neg()
        oloss = torch.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = torch.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()
