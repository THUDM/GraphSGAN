# GraphSGAN
Implementation of "GraphSGAN", a GAN-based semi-supervised learning algorithm for graph data.

Paper: [Semi-supervised Learning on Graphs with Generative Adversarial Nets](https://arxiv.org/abs/1809.00130)

## Preparation

> unzip cora.dataset.zip

The codes are written under `Python==2.7` and `pytorch~=3`. If you want to run it in other environments, minor changes might be needed.

## Run 

> python GraphSGAN.py --cuda

will run a example, banchmark cora task.

The programme takes `FeatureGraphDataset` as input and `cora.dataset` is built from `FeatureGraphDataset.py`. You can build your own FeatureGraphDataset.

`Early stop` and tuned hyperparameters are not included in this minimal release. You can determine them based on your own validation set.

> rm -rf models logfile

can delete the saved models and logfile to retrain.

You can visualize the infomation in logfile by using `tensorboard`.

The expected accuracy of example should be above `0.83`.

## How to run GraphSGAN on other datasets?
1. Build `FeatureGraphDataset` for new dataset.
  This class takes init parameters as below:
```
    features: numpy ndarray, [[f1, f2, ...], [f1, f2, ...]]
    label: numpy ndarray, [0, 1, 2, 0, ...]
    adj: dict of (int, list of int), {[1,2],[0,3],...}
```
 2. Load embeddings for dataset.
 
   It is recommended to use `read_embeddings` method to read embeddings from file.
   
   The first line of embeddings file are two integers: n and dim.
   
   In the next n lines, each line contains dim + 1 integers. The first is the No. of the node and the rest are embeddings.
   
   Example:
```
  3 2
  0 0.123 0.233
  1 0.720 -0.121
  2 0.778 -0.921
  3 0.161 -0.775
```
 3. Setting splits
 
  call `setting(label_num_per_class, test_num)`
  
 4. Replace `dataset` in `GraphSGAN.py` with built new dataset.
 
 # Performance on some other datasets

In the paper of [SCAN_DIS](https://papers.nips.cc/paper/8878-semi-supervisedly-co-embedding-attributed-networks.pdf), the performance of GraphSGAN on `Pubmed, Flickr` and `BlogCatalog` are tested:

<i></i>| Pubmed | <i></i> | <i></i> | Flickr | <i></i> | <i></i> | BlogCatalog | <i></i> | 
--- | --- | --- | --- |--- |--- |--- |--- |--- |
Marco-F1 | Micro-F1 | Acc | Marco-F1 | Micro-F1 | Acc | Marco-F1 | Micro-F1 | Acc | 
.839 | .842 | .841 | .697 | .715 | .702 | .698 | .703 | .719 |

Although not responsible for the results, we think it is really worth reference.
