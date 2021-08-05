# Transformer Implementations

<p>
  <a href="https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/UdbhavPrasad072300/Transformer-Implementations">
  </a>
  <a href="https://pypi.org/project/transformer-implementations/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/v/transformer-implementations">
  </a>
  <a href="https://pypi.org/project/transformer-implementations/">
        <img alt="PyPi Downloads" src="https://img.shields.io/pypi/dm/transformer-implementations">
  </a>
  <a href="https://pypi.org/project/transformer-implementations/">
        <img alt="Package Status" src="https://img.shields.io/pypi/status/transformer-implementations">
  </a>
</p>

Transformer Implementations and some examples with them

Implemented:
<ul>
  <li>Vanilla Transformer</li>
  <li>ViT - Vision Transformers</li>
  <li>DeiT - Data efficient image Transformers</li>
  <li>BERT - Bidirectional Encoder Representations from Transformers</li>
  <li>GPT - Generative Pre-trained Transformer</li>
</ul>

## Installation

<a href="https://pypi.org/project/transformer-implementations/">PyPi</a>

```bash
$ pip install transformer-implementations
```

or

```bash
python setup.py build
python setup.py install
```

## Example

In <a href="https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/notebooks/">notebooks</a> directory there is a notebook on how to use each of these models for their intented use; such as image classification for Vision Transformer (ViT) and others.
Check them out!

```python
from transformer_package.models import ViT

image_size = 28 # Model Parameters
channel_size = 1
patch_size = 7
embed_size = 512
num_heads = 8
classes = 10
num_layers = 3
hidden_size = 256
dropout = 0.2

model = ViT(image_size, 
            channel_size, 
            patch_size, 
            embed_size, 
            num_heads, 
            classes, 
            num_layers, 
            hidden_size, 
            dropout=dropout).to(DEVICE)
            
prediction = model(image_tensor)
```

## Language Translation

from "Attention is All You Need": https://arxiv.org/pdf/1706.03762.pdf

Models trained with Implementation:
<ul>
  <li><a href="https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/notebooks/Multi30k%20-%20Language%20Translation.ipynb">Multi30k - German to English</a></li>
</ul>

## Multi-class Image Classification with Vision Transformers (ViT)

from "An Image is Worth 16x16 words: Transformers for image recognition at scale": https://arxiv.org/pdf/2010.11929v1.pdf

Models trained with Implementation:
<ul>
 <li><a href="https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/notebooks/MNIST%20Classification%20-%20ViT.ipynb">MNIST - Grayscale Images</a></li>
  <li><a href="https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/notebooks/CIFAR10%20Classification%20-%20ViT.ipynb">CIFAR10 - MultiChannel Images</a></li>
</ul>

Note: ViT will not perform great on small datasets

## Multi-class Image Classification with Data-efficient image Transformers (DeiT)

from "Training data-efficient image transformers & distillation through attention": https://arxiv.org/pdf/2012.12877v1.pdf

Models trained with Implementation:
<ul>
 <li><a href="https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/pre-train/VGG16_CIFAR10.ipynb">Pretraining Teacher model for Distillation</a></li>
 <li><a href="https://github.com/UdbhavPrasad072300/Transformer-Implementations/blob/main/notebooks/CIFAR10%20Classification%20-%20DeiT.ipynb">CIFAR10 - Low Res Images</a></li>
</ul>
