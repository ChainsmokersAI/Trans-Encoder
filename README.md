# Iterative Knowledge Distillation between Bi/Cross-Encoder (Trans-Encoder)
From pre-trained ***unsupervised bi-encoder*** (SimCSE), one can boost the performance of both *bi-* and *cross-encoder* through iterative knowledge distillation process between two models.<br/>
Just using sentence pairs of [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) train set (**without labels**), distilled *bi-encoder* shows an extra performance gain on same test set.<br/>
I have referenced the following [Amazon's paper](https://arxiv.org/abs/2109.13059), but codes are **NOT exactly same** as in paper.
* Trans-Encoder: Unsupervised sentence-pair modelling through self- and mutual-distillations ([Liu et al.](https://arxiv.org/abs/2109.13059), 2021)

Also, [Korean review](https://chainsmokers.oopy.io/paper/trans-encoder) is on my own blog.
## Usage (OS: Ubuntu)
### Dependencies
* pandas
* numpy
* scipy
* matplotlib
* torch (1.11.0)
* transformers (4.18.0)
* tensorboard
### Initialization
```bash
git clone https://github.com/ChainsmokersAI/Trans-Encoder.git
cd Trans-Encoder/
sh download_dataset.sh
```
### Notebooks
Training and evaluation codes are written in Jupyter Notebook (.ipynb) files.<br/>
The codes are in [/notebook](https://github.com/ChainsmokersAI/Trans-Encoder/tree/main/notebook) directory.
* Fine-Tune Cross-Encoder with Ground-Truth Labels [[codes](https://github.com/ChainsmokersAI/Trans-Encoder/blob/main/notebook/1.%20Train%20Cross-Encoder%20with%20Ground-Truth%20Labels.ipynb)]
* Bi-to-Cross Encoder Distillation [[codes](https://github.com/ChainsmokersAI/Trans-Encoder/blob/main/notebook/2.%20Create%20Pseudo-Labels%20using%20Bi-Encoder%20%26%20Train%20Cross-Encoder.ipynb)]
* Cross-to-Bi Encoder Distillation [[codes](https://github.com/ChainsmokersAI/Trans-Encoder/blob/main/notebook/3.%20Create%20Pseudo-Labels%20using%20Cross-Encoder%20%26%20Train%20Bi-Encoder.ipynb)]
* **Full (Self-)Distillation** [[codes](https://github.com/ChainsmokersAI/Trans-Encoder/blob/main/notebook/4.%20Trans-Encoder%20(Distillation%20Loop).ipynb)]
### Results
Experiments were conducted on single GeForce RTX 3090 GPU.<br/>
All distilled models were evaluated on [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) test set (Spearmanr).
|Iteration|Direction|Cross-Encoder|Bi-Encoder|
|:---:|:---:|:---:|:---:|
|0|||80.10|
|1|bi->cross|83.07||
|1|cross->bi||80.98|
|2|bi->cross|83.44||
|2|cross->bi||81.01|
|3|bi->cross|**83.56**||
|3|cross->bi||**81.02**|
