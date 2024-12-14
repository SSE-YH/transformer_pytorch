import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from transformers import BertTokenizer


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.dcoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

