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
