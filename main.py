import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import Transformer, PositionalEncoding

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used: {}".format(device))

    # Model Parameters
    source_vocab_size = 1000
    target_vocab_size = 1200
    embed_size = 30
    num_head = 3
    num_ff = 300
    encoder_layers = 1
    decoder_layers = 1

    model = Transformer(source_vocab_size, target_vocab_size, embed_size, num_head, num_ff, encoder_layers, decoder_layers, device=device).to(device)

    print("-" * 100)
    print(model)
    print("-" * 100)

    x = torch.rand(64, 500).type(torch.LongTensor).to(device)
    y = torch.rand(64, 400).type(torch.LongTensor).to(device)

    print("Input Dimensions: {} & {}".format(x.size(), y.size()))

    out = model(x, y)

    print("Output Dimensions: {}".format(out.size()))

    print("Program has Ended")