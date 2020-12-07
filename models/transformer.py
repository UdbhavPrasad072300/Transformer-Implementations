import torch
import torch.nn as nn
import torch.nn.functional as F

import math


# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=2000, embedding_size=300, dropout=0.2, device="cpu"):
        super(PositionalEncoding, self).__init__()
        import math

        self.dropout = nn.Dropout(p=dropout)

        self.pe_matrix = torch.zeros(max_seq_len, embedding_size).to(device)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        self.pe_matrix[:, 0::2] = torch.sin(position * div_term)
        self.pe_matrix[:, 1::2] = torch.cos(position * div_term)

        self.pe_matrix = self.pe_matrix.unsqueeze(0).transpose(0, 1)

        self.register_buffer("Positional Encoding", self.pe_matrix)

    def forward(self, x):
        x = x + self.pe_matrix[:x.size(0), :]
        x = self.dropout(x)
        return x


class Transformer(nn.Module):
    def __init__(self, s_vocab_size, t_vocab_size, embed_size, num_head, num_ff, encode_layers, decode_layers,
                 dropout=0.2, device="cpu"):
        super(Transformer, self).__init__()

        self.s_vocab_size = s_vocab_size
        self.t_vocab_size = t_vocab_size
        self.embed_size = embed_size
        self.num_head = num_head
        self.num_ff = num_ff
        self.encoder_num_layers = encode_layers
        self.decoder_num_layers = decode_layers
        self.dropout = dropout
        self.device = device

        self.encoder_embed = nn.Embedding(self.s_vocab_size, embed_size)
        self.decoder_embed = nn.Embedding(self.t_vocab_size, embed_size)
        self.encoder_positional_encoding = PositionalEncoding(self.s_vocab_size, self.embed_size, device=device)
        self.decoder_positional_encoding = PositionalEncoding(self.t_vocab_size, self.embed_size, device=device)

        self.encoder_layer = nn.TransformerEncoderLayer(self.embed_size, self.num_head, self.num_ff,
                                                        dropout=self.dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, self.encoder_num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(self.embed_size, self.num_head, self.num_ff,
                                                        dropout=self.dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, self.decoder_num_layers)

        self.final = nn.Linear(self.embed_size, self.t_vocab_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x, y):
        x = self.encoder_embed(x) * math.sqrt(self.embed_size)
        y = self.decoder_embed(y) * math.sqrt(self.embed_size)

        x = self.encoder_positional_encoding(x)
        y = self.decoder_positional_encoding(y)

        memory = self.encoder(x)

        out = self.decoder(y, memory)

        x = self.final(out)
        return x
