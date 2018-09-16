# -*- coding:utf-8 -*-
from __future__ import print_function 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from functional import log_sum_exp, pull_away_term
from torch.utils.data import DataLoader,TensorDataset
import sys
from torch.nn.parameter import Parameter
import argparse
from Nets import Generator, Discriminator
import tensorboardX
import os
import random
from FeatureGraphDataset import FeatureGraphDataset
import pickle as pkl
class GraphSGAN(object):
    def __init__(self, G, D, dataset, args):
        if os.path.exists(args.savedir):
            print('Loading model from ' + args.savedir)
            self.G = torch.load(os.path.join(args.savedir, 'G.pkl'))
            self.D = torch.load(os.path.join(args.savedir, 'D.pkl'))
            self.embedding_layer = torch.load(os.path.join(args.savedir, 'embedding.pkl'))
        else:
            os.makedirs(args.savedir)
            self.G = G
            self.D = D
            self.embedding_layer = nn.Embedding(dataset.n, dataset.d)
            self.embedding_layer.weight = Parameter(torch.Tensor(dataset.embbedings))
            torch.save(self.G, os.path.join(args.savedir, 'G.pkl'))
            torch.save(self.D, os.path.join(args.savedir, 'D.pkl'))
            torch.save(self.embedding_layer, os.path.join(args.savedir, 'embedding.pkl'))

        self.writer = tensorboardX.SummaryWriter(log_dir=args.logdir)
        if args.cuda:
            self.G.cuda()
            self.D.cuda() # self.embedding_layer is on CPU
        self.dataset = dataset
        self.Doptim = optim.Adam(self.D.parameters(), lr=args.lr, betas= (args.momentum, 0.999))
        self.Goptim = optim.Adam(self.G.parameters(), lr=args.lr, betas = (args.momentum,0.999))
        self.args = args

    def trainD(self, idf_label, y, idf_unlabel):
        x_label, x_unlabel, y = self.make_input(*idf_label), self.make_input(*idf_unlabel), Variable(y, requires_grad = False)
        if self.args.cuda:
            x_label, x_unlabel, y = x_label.cuda(), x_unlabel.cuda(), y.cuda()
        output_label, (mom_un, output_unlabel), output_fake = self.D(x_label, cuda=self.args.cuda), self.D(x_unlabel, cuda=self.args.cuda, feature = True), self.D(self.G(x_unlabel.size()[0], cuda = self.args.cuda).view(x_unlabel.size()).detach(), cuda=self.args.cuda)
        logz_label, logz_unlabel, logz_fake = log_sum_exp(output_label), log_sum_exp(output_unlabel), log_sum_exp(output_fake) # log ∑e^x_i
        prob_label = torch.gather(output_label, 1, y.unsqueeze(1)) # log e^x_label = x_label 
        loss_supervised = -torch.mean(prob_label) + torch.mean(logz_label)
        loss_unsupervised = 0.5 * (-torch.mean(logz_unlabel) + torch.mean(F.softplus(logz_unlabel))  + # real_data: log Z/(1+Z)
                            torch.mean(F.softplus(logz_fake)) ) # fake_data: log 1/(1+Z)
        entropy = -torch.mean(F.softmax(output_unlabel, dim = 1) * F.log_softmax(output_unlabel, dim = 1))
        pt = pull_away_term(mom_un)
        loss = loss_supervised + self.args.unlabel_weight * loss_unsupervised + entropy + pt
        acc = torch.mean((output_label.max(1)[1] == y).float())
        self.Doptim.zero_grad()
        loss.backward()
        self.Doptim.step()
        return loss_supervised.data.cpu().numpy(), loss_unsupervised.data.cpu().numpy(), acc
    
    def trainG(self, idf_unlabel):
        x_unlabel = self.make_input(*idf_unlabel)
        if self.args.cuda:
            x_unlabel = x_unlabel.cuda()
        fake = self.G(x_unlabel.size()[0], cuda = self.args.cuda).view(x_unlabel.size())
        mom_gen, output_fake = self.D(fake, feature=True, cuda=self.args.cuda)
        mom_unlabel, output_unlabel = self.D(x_unlabel, feature=True, cuda=self.args.cuda)
        loss_pt = pull_away_term(mom_gen)
        mom_gen = torch.mean(mom_gen, dim = 0)
        mom_unlabel = torch.mean(mom_unlabel, dim = 0) 
        loss_fm = torch.mean(torch.abs(mom_gen - mom_unlabel))
        loss = loss_fm + loss_pt 
        self.Goptim.zero_grad()
        self.Doptim.zero_grad()
        loss.backward()
        self.Goptim.step()
        return loss.data.cpu().numpy()

    def make_input(self, ids, feature, volatile = False):
        '''Concatenate feature and embeddings

        Args:
            feature: Size=>[batch_size, dataset.k], Type=>FloatTensor
            ids: Size=>[batch_size], Type=>LongTensor
        '''
        embedding = self.embedding_layer(Variable(ids, volatile = volatile)).detach() # detach temporarily
        return torch.cat((Variable(feature), embedding), dim = 1)
    def train(self):
        gn = 0
        NUM_BATCH = 100
        for epoch in range(self.args.epochs):
            self.G.train()
            self.D.train()
            self.D.turn = epoch
            loss_supervised = loss_unsupervised = loss_gen = accuracy = 0.
            for batch_num in range(NUM_BATCH):
                # extract batch from dataset
                idf_unlabel1 = self.dataset.unlabel_batch(self.args.batch_size)
                idf_unlabel2 = self.dataset.unlabel_batch(self.args.batch_size)
                id0, xf, y = self.dataset.label_batch(self.args.batch_size)

                # train D
                ll, lu, acc = self.trainD((id0, xf), y, idf_unlabel1)
                loss_supervised += ll
                loss_unsupervised += lu
                accuracy += acc

                # train G on unlabeled data
                lg = self.trainG(idf_unlabel2)
                loss_gen += lg
                # print and record logs 
                if (batch_num + 1) % self.args.log_interval == 0:
                    print('Training: %d / %d' % (batch_num + 1, NUM_BATCH))
                    gn += 1
                    self.writer.add_scalars('loss', {'loss_supervised':ll, 'loss_unsupervised':lu, 'loss_gen':lg}, gn)
                    self.writer.add_histogram('real_feature', self.D(self.make_input(id0, xf, volatile = True).cuda(), cuda=self.args.cuda, feature = True)[0], gn)
                    self.writer.add_histogram('fake_feature', self.D(self.G(self.args.batch_size, cuda = self.args.cuda), cuda=self.args.cuda, feature = True)[0], gn)
            # calculate average loss at the end of an epoch
            batch_num += 1
            loss_supervised /= batch_num
            loss_unsupervised /= batch_num
            loss_gen /= batch_num
            accuracy /= batch_num
            print("Iteration %d, loss_supervised = %.4f, loss_unsupervised = %.4f, loss_gen = %.4f train acc = %.4f" % (epoch, loss_supervised, loss_unsupervised, loss_gen, accuracy))
            sys.stdout.flush()

            # eval
            tmp = self.eval()
            print("Eval: correct %d / %d, Acc: %.2f"  % (tmp, self.dataset.test_num, tmp * 100. / self.dataset.test_num))
            torch.save(self.G, os.path.join(self.args.savedir, 'G.pkl'))
            torch.save(self.D, os.path.join(self.args.savedir, 'D.pkl'))


    def predict(self, x):
        '''predict label in volatile mode

        Args:
            x: Size=>[batch_size, self.dataset.k + self.dataset.d], Type=>Variable(FloatTensor), volatile
        '''
        return torch.max(self.D(x, cuda=self.args.cuda), 1)[1].data

    def eval(self):
        self.G.eval()
        self.D.eval()
        ids, f, y = self.dataset.test_batch()
        x = self.make_input(ids, f, volatile = True)
        if self.args.cuda:
            x, y = x.cuda(), y.cuda()
        pred1 = self.predict(x)

        return torch.sum(pred1 == y)

    def draw(self, batch_size):
        self.G.eval()
        return self.G(batch_size, cuda=self.args.cuda)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch GraphS GAN')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before evaling training status')
    parser.add_argument('--unlabel-weight', type=float, default=0.5, metavar='N',
                        help='scale factor between labeled and unlabeled data')
    parser.add_argument('--logdir', type=str, default='./logfile', metavar='LOG_PATH', help='logfile path, tensorboard format')
    parser.add_argument('--savedir', type=str, default='./models', metavar='SAVE_PATH', help = 'saving path, pickle format')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # That is how you usually build the dataset 
    #dataset = CoraDataset(feature_file = './data/cora.features', 
    #     edge_file = './data/cora_edgelist', label_file = './data/cora_label')
    #dataset.read_embbedings('./embedding/embedding_line_cora')
    #dataset.setting(20, 1000)
    

    # but we load the example of cora
    with open('cora.dataset', 'r') as fdata:
        dataset = pkl.load(fdata)
    gan = GraphSGAN(Generator(200, dataset.k + dataset.d), Discriminator(dataset.k + dataset.d, dataset.m), dataset, args)
    gan.train() 
    
