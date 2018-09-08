#!usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function
import sys

from models.ConvNet import * 
from layers import *
from data_utils import *
from utils import *
ctx = check_ctx()

from optim import *

# importing MxNet >= 1.0
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd, gluon

import random

mx.random.seed(1)
random.seed(1)


def evaluate_accuracy(data_iterator, num_examples, batch_size, params, net, pool_type,pool_size,pool_stride):
    numerator = 0.
    denominator = 0.
    for batch_i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((data.shape[0],1,1,-1))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, 10)
        output, _ = net(data, params,pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
        print('Evaluating accuracy. (complete percent: %.2f/100' %(1.0 * batch_i / (num_examples//batch_size) * 100) +')' , end='')
        sys.stdout.write("\r")
    return (numerator / denominator).asscalar()


def Solver(train, test, Debug, batch_size, lr
          , smoothing_constant, num_fc1, num_fc2, num_outputs, epochs, SNR
          , sl, pool_type ,pool_size ,pool_stride, params_init=None, period=None):
    
    num_examples = train.shape[0]
    # 训练集数据类型转换
    y = nd.array(~train.sigma.isnull() +0)
    X = nd.array(Normolise(train.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
    print('Label for training:', y.shape)
    print('Dataset for training:', X.shape, end='\n\n')

    dataset_train = gluon.data.ArrayDataset(X, y)
    train_data = gluon.data.DataLoader(dataset_train, batch_size, shuffle=True, last_batch='keep')

    y = nd.array(~test.sigma.isnull() +0)
    X = nd.array(Normolise(test.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
    print('Label for testing:', y.shape)
    print('Dataset for testing:', X.shape, end='\n\n')
    
    # 这里使用data模块来读取数据。创建测试数据。  (suffle)
    dataset_test = gluon.data.ArrayDataset(X, y)
    test_data = gluon.data.DataLoader(dataset_test, batch_size, shuffle=True, last_batch='keep')

    
    # Train
    loss_history = []
    loss_v_history = []
    moving_loss_history = []
    test_accuracy_history = []
    train_accuracy_history = []
    
#     assert period >= batch_size and period % batch_size == 0
    
    # Initializate parameters
    if params_init:
        print('Loading params...')
        params = params_init

        [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6] = params

        # random fc layers
        weight_scale = .01
        W7 = nd.random_normal(loc=0, scale=weight_scale, shape=(sl, num_fc1), ctx=ctx )
        W8 = nd.random_normal(loc=0, scale=weight_scale, shape=(num_fc1, num_fc2), ctx=ctx )        
        W9 = nd.random_normal(loc=0, scale=weight_scale, shape=(num_fc2, num_outputs), ctx=ctx )
        b7 = nd.random_normal(shape=num_fc1, scale=weight_scale, ctx=ctx)
        b8 = nd.random_normal(shape=num_fc2, scale=weight_scale, ctx=ctx)    
        b9 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)  

        params = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6]
        print('Random the FC1&2-layers...')

        vs = []
        sqrs = [] 
        for param in params:
            param.attach_grad()
            vs.append(param.zeros_like())
            sqrs.append(param.zeros_like())              
    else:
        params, vs, sqrs = init_params(num_fc1 = 64, num_fc2 = 64, num_outputs = 2, sl=sl)
        print('Initiate weights from random...')

    # Debug
    if Debug:
        print('Debuging...')
        if params_init:
            params = params_init
        else:
            params, vs, sqrs = init_params(num_fc1 = 64, num_fc2 = 64, num_outputs = 2, sl=sl)
        for data, _ in train_data:
            data = data.as_in_context(ctx).reshape((batch_size,1,1,-1))
            break
        print(pool_type, pool_size, pool_stride)
        _, _ = ConvNet(data, params, debug=Debug, pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        print()
    
#     total_loss = [Total_loss(train_data_10, params, batch_size, num_outputs)]
    
    t = 0
#   Epoch starts from 1.
    print('pool_type: ', pool_type)
    print('pool_size: ', pool_size)
    print('pool_stride: ', pool_stride)
    print('sl: ', sl)
    for epoch in range(1, epochs + 1):
        Epoch_loss = []
#         学习率自我衰减。
        if epoch > 2:
#             lr *= 0.1
            lr /= (1+0.01*epoch)
        for batch_i, ((data, label),(data_v, label_v)) in enumerate(zip(train_data, test_data)):
            data = data.as_in_context(ctx).reshape((data.shape[0],1,1,-1))
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, num_outputs)
            with autograd.record():
                output, _ = ConvNet(data, params, pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
#             print(output)
            # params = sgd(params, lr, batch_size)

#           Increment t before invoking adam.
            t += 1
            params, vs, sqrs = adam(params, vs, sqrs, lr, batch_size, t)

            data_v = data_v.as_in_context(ctx).reshape((data_v.shape[0],1,1,-1))
            label_v = label_v.as_in_context(ctx)
            label_v_one_hot = nd.one_hot(label_v, num_outputs)
            output_v, _ = ConvNet(data_v, params, pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
            loss_v = softmax_cross_entropy(output_v, label_v_one_hot)            
            
#             #########################
#              Keep a moving average of the losses
#             #########################
            curr_loss = nd.mean(loss).asscalar()
            curr_loss_v = nd.mean(loss_v).asscalar()
            moving_loss = (curr_loss if ((batch_i == 0) and (epoch-1 == 0))
                           else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

            loss_history.append(curr_loss)
            loss_v_history.append(curr_loss_v)
            moving_loss_history.append(moving_loss)
            Epoch_loss.append(curr_loss)
#             if batch_i * batch_size % period == 0:
#                 print('Curr_loss: ', curr_loss)
                
            print('Working on epoch %d. Curr_loss: %.5f (complete percent: %.2f/100' %(epoch, curr_loss*1.0, 1.0 * batch_i / (num_examples//batch_size) * 100) +')' , end='')
            sys.stdout.write("\r")
            # print('{"metric": "Training Loss for ALL", "value": %.5f}' %(curr_loss*1.0) )
            # print('{"metric": "Testing Loss for ALL", "value": %.5f}' %(curr_loss_v*1.0) )
#             print('{"metric": "Training Loss for SNR=%s", "value": %.5f}' %(str(SNR), curr_loss*1.0) )
#             print('{"metric": "Testing Loss for SNR=%s", "value": %.5f}' %(str(SNR), curr_loss_v*1.0) )
        train_accuracy = evaluate_accuracy(train_data, num_examples, batch_size, params, ConvNet,pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        test_accuracy = evaluate_accuracy(test_data, num_examples, batch_size, params, ConvNet,pool_type=pool_type,pool_size = pool_size,pool_stride=pool_stride)
        test_accuracy_history.append(test_accuracy)
        train_accuracy_history.append(train_accuracy)


        print("Epoch %d, Moving_loss: %.6f, Epoch_loss(mean): %.6f, Train_acc %.4f, Test_acc %.4f" %
              (epoch, moving_loss, np.mean(Epoch_loss), train_accuracy, test_accuracy))
#         print('{"metric": "Train_acc. for SNR=%s in epoches", "value": %.4f}' %(str(SNR), train_accuracy) )
#         print('{"metric": "Test_acc. for SNR=%s in epoches", "value": %.4f}' %(str(SNR), test_accuracy) )
        yield (params, loss_history, loss_v_history, moving_loss_history, test_accuracy_history, train_accuracy_history)
    
    
    
def predict(data, net, params):

    X = nd.array(Normolise(data.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
    # num_examples = data.shape[0]
    # batch_size = [2**i for i in range(10) if num_examples%(2**i) == 0][-1]
    # print('Batch size = %s' %batch_size)
    data = nd.array(X).as_in_context(ctx).reshape((-1,1,1,8192))
    output, interlayer = net(data, params)
    prob = transform_softmax(output)[:,1].asnumpy().tolist()[0]
    return prob, interlayer



    
def predict_(data, net, params):

    X = nd.array(Normolise(data))
    # num_examples = data.shape[0]
    # batch_size = [2**i for i in range(10) if num_examples%(2**i) == 0][-1]
    # print('Batch size = %s' %batch_size)
    data = nd.array(X).as_in_context(ctx).reshape((-1,1,1,8192))
    output, _ = net(data, params)
    prob = transform_softmax(output)[:,1].asnumpy().tolist()[0]
    return prob, output



if __name__ == '__main__':
    print('CPU or GPU? : ', ctx)
    
    
    
class Solver_nd(object):
    
    def __init__(self, model, train, test, SNR, **kwargs):
        self.model = model
        self.num_channel = model.input_dim[0]
        self.train = train
        self.test = test
        try:
            assert self.train.shape == self.test.shape
        except:
            print('self.train.shape != self.test.shape')
        
        self.SNR = SNR

#         self.update_rule = kwargs.pop('update_rule', 'sgd')
#         self.optim_config = kwargs.pop('optim_config', {})
        self.params = kwargs.pop('params', None)   # Transfer learning
        if self.params:  # 若有迁移学习
            self.params = self.params.copy()
            try:         # 考察导入的模型参数变量 与 导入模型的参数之间得到关系
                assert (self.params == model.params)# and (self.params is not model.params)
            except Exception as e:
                print(e)
                print('导入的模型参数与导入模型现默认参数有着相同的值~')
        self.batch_size = kwargs.pop('batch_size', 256)
        self.lr_rate = kwargs.pop('lr_rate', 0.01)
        self.lr_decay = kwargs.pop('lr_decay', 0.01)
        self.num_epoch = kwargs.pop('num_epoch', 10)
        self.smoothing_constant = kwargs.pop('smoothing_constant', 0.01)
        
        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
        self.verbose = kwargs.pop('verbose', False)
        self.oldversion = kwargs.pop('oldversion', False)
#         self.print_every = kwargs.pop('print_every', 100)
        

        if len(kwargs) != 0:
            extra = ', '.join('"%s"' %k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)
            
#         if not hasattr(optim, self.update_rule):
#             raise ValueError('Unrecognized update rule: "%s"' % self.update_rule)
#         self.update_rule = getattr(optim, self.update_rule)
        
        if self.oldversion:
            self._reset_data_old()
        else:
            self._reset_data()
        
        if self.params:
            self._reset_params_Transfer()
        else:
            self._reset_params()
        
        
        
    def _reset_data_old(self):
        try:
            assert self.train.shape[1] == self.test.shape[1]
        except:
            print('self.train.shape[1] != self.test.shape[1],',self.train.shape[1],self.test.shape[1])
        self.train_size = self.train.shape[0]
        self.test_size = self.test.shape[0]
        
        y = nd.array(~self.train.sigma.isnull() +0)
        X = nd.array(Normolise(self.train.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
        print('Label for training:', y.shape)
        print('Dataset for training:', X.shape, end='\n\n')

        dataset_train = gluon.data.ArrayDataset(X, y)
        self.train_data = gluon.data.DataLoader(dataset_train, self.batch_size, shuffle=True, last_batch='keep')

        y = nd.array(~self.test.sigma.isnull() +0)
        X = nd.array(Normolise(self.test.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
        print('Label for testing:', y.shape)
        print('Dataset for testing:', X.shape, end='\n\n')

        dataset_test = gluon.data.ArrayDataset(X, y)
        self.test_data = gluon.data.DataLoader(dataset_test, self.batch_size, shuffle=True, last_batch='keep')        
        

    def _reset_data(self):
        
        try:
            assert self.train.shape[1] == self.test.shape[1]
        except:
            print('self.train.shape[1] != self.test.shape[1]')
        
        self.train_size = self.train.shape[0]
        self.test_size = self.test.shape[0]
        noiseAll_size = self.train_size+self.test_size

        self.param_noise = Pre_zero(size = (noiseAll_size,) + (self.train.shape[1:]))

        self.y_train = nd.concat(nd.ones(shape = (self.train_size,), ctx = ctx), nd.zeros(shape = (self.train_size,), ctx = ctx) , dim = 0)
        self.y_test = nd.concat(nd.ones(shape = (self.test_size,), ctx = ctx), nd.zeros(shape = (self.test_size,), ctx = ctx) , dim = 0)
        print('Label for training:', self.y_train.shape)
        print('Label for testing:', self.y_test.shape)

    def _reset_params_Transfer(self):
        self.epoch = 0
        self.best_test_acc = 0
        self.best_params = {}
        self.moving_loss = 0

        self.train_acc_history = []
        self.test_acc_history = []

        self.loss_history = []
        self.loss_v_history = []
        self.moving_loss_history = []
            
#         self.optim_configs = {}
#         for p in self.model.params:
#             d = {k: v for k, v in self.optim_config.items()}
#             self.optim_configs[p] = d    


        # Opt. for Adam ############
        self.vs = []
        self.sqrs = []
        
        # Transfer Learning ########
        self.model.init_params()
        for key, params in self.params.items():
            if params.shape[0] == self.model.flatten_dim:
                break
            self.model.params[key] = params.copy()

        # And assign space for gradients
        for param in self.model.params.values():
            param.attach_grad()
            self.vs.append(param.zeros_like())
            self.sqrs.append(param.zeros_like())        

    def _reset_params(self):
        self.epoch = 0
        self.best_test_acc = 0
        self.best_params = {}
        self.moving_loss = 0

        self.train_acc_history = []
        self.test_acc_history = []

        self.loss_history = []
        self.loss_v_history = []
        self.moving_loss_history = []
            
#         self.optim_configs = {}
#         for p in self.model.params:
#             d = {k: v for k, v in self.optim_config.items()}
#             self.optim_configs[p] = d    


        # Opt. for Adam ############
        self.vs = []
        self.sqrs = []

        # And assign space for gradients
        for param in self.model.params.values():
            param.attach_grad()
            self.vs.append(param.zeros_like())
            self.sqrs.append(param.zeros_like())

        


    def Training(self, Iterator = False):
        
        t = 0    
        try:

            for epoch in range(1, self.num_epoch + 1):
                self.epoch = epoch
                self.lr_rate = lr_decay(self.lr_rate, epoch, self.lr_decay)

                if self.oldversion: pass
                else: self._reset_noise()

                self._iteration(t, epoch)

                self.train_acc_history.append(self.check_acc(self.train_data))
                val_acc = self.check_acc(self.test_data)
                self.test_acc_history.append(val_acc)
                self._save_checkpoint()


                if val_acc > self.best_test_acc:
                    self.best_test_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

                if self.verbose:
                    pass
    #                 print('{"metric": "Train_acc. for SNR=%s in epoches", "value": %.4f}' %(str(SNR), train_accuracy) )
    #                 print('{"metric": "Test_acc. for SNR=%s in epoches", "value": %.4f}' %(str(SNR), test_accuracy) )
                else:
                    print("Epoch {:d}, Moving_loss: {:.6f}, Epoch_loss(mean): {:.6f}, Train_acc {:.4f}, Test_acc {:.4f}(Best:{:.4f})".format(epoch, self.moving_loss_history[-1], np.mean(self.Epoch_loss), self.train_acc_history[-1], self.test_acc_history[-1], self.best_test_acc))
#                 if Iterator:
#                     yield self.loss_history, self.loss_v_history, self.moving_loss_history, self.train_acc_history, self.test_acc_history

        except KeyboardInterrupt as e:
            print(e)
            print('Early stoping at epoch=%s' %str(epoch))        

        self.model.params = self.best_params
        print('Finished!')

    def _iteration(self, t, epoch):
        
        self.Epoch_loss = []

        
        for batch_i, ((data, label),(data_v, label_v)) in enumerate(zip(self.train_data, self.test_data)):

            loss = self.loss(data, label, train = True)
            loss_v, _= self.loss(data_v, label_v, train = False)

            # Increment t before invoking adam.
            t += 1
            self.model.params, self.vs, self.sqrs = adam(self.model.params, self.vs, self.sqrs, self.lr_rate, self.batch_size, t)
        
            # Keep a moving average of the losses
            curr_loss = nd.mean(loss).asscalar()
            curr_loss_v = nd.mean(loss_v).asscalar()
            self.moving_loss = (curr_loss if ((batch_i == 0) and (epoch-1 == 0))
                           else (1 - self.smoothing_constant) * self.moving_loss + (self.smoothing_constant) * curr_loss)

            self.loss_history.append(curr_loss)
            self.loss_v_history.append(curr_loss_v)
            self.moving_loss_history.append(self.moving_loss)
            self.Epoch_loss.append(curr_loss)

            if self.verbose:
                pass
            # print('{"metric": "Training Loss for ALL", "value": %.5f}' %(curr_loss*1.0) )
            # print('{"metric": "Testing Loss for ALL", "value": %.5f}' %(curr_loss_v*1.0) )
#             print('{"metric": "Training Loss for SNR=%s", "value": %.5f}' %(str(SNR), curr_loss*1.0) )
#             print('{"metric": "Testing Loss for SNR=%s", "value": %.5f}' %(str(SNR), curr_loss_v*1.0) )            
            else:
                print('Working on epoch {:d}. Curr_loss: {:.5f} (complete percent: {:.2f}/100)'.format(epoch, curr_loss*1.0, 1.0 * batch_i / (self.train_size/self.batch_size) * 100/ 2) , end='')
                sys.stdout.write("\r")



    def loss(self, data, label, train=True):
        data = data.as_in_context(ctx).reshape((data.shape[0],self.num_channel,1,-1))
        label = label.as_in_context(ctx)
        label_one_hot = nd.one_hot(label, self.model.output_dim)
        
        if train:
            with autograd.record():
                output, _= self.model.network(X=data)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
            return loss
        else:
            output, _ = self.model.network(X=data)
            loss = softmax_cross_entropy(output, label_one_hot)
            return loss, output
        
        

    def gen_noise(self):
        
        if ctx == mx.gpu():
            noise, _ = TimeseriesFromPSD_nd(self.param_noise)
        elif ctx == mx.cpu():
            noise, _ = TimeseriesFromPSD(self.param_noise)
            noise = nd.array(noise)
            
        return noise

    

    def _reset_noise(self):
        
        # noise for mixing
        noise = self.gen_noise()
        
        sigma = self.train.max(axis = self.num_channel) / float(self.SNR) / nd.array(noise[:self.train_size].asnumpy().std(axis = self.num_channel,dtype='float64'),ctx=ctx)
        self.sigma = sigma
        signal_train = nd.divide(self.train, sigma.reshape((-1,self.num_channel,1)))
        data_train = signal_train + noise[:self.train_size]
        
        sigma = self.test.max(axis = self.num_channel) / float(self.SNR) / nd.array(noise[-self.test_size:].asnumpy().std(axis = self.num_channel,dtype='float64'),ctx=ctx)
        signal_test = nd.divide(self.test, sigma.reshape((-1,self.num_channel,1)))    
        data_test = signal_test + noise[-self.test_size:]
        
        # noise for pure conterpart
        noise = self.gen_noise()
        
        X_train = Normolise_nd(nd.concat(data_train, noise[:self.train_size], dim=0), self.num_channel)
        dataset_train = gluon.data.ArrayDataset(X_train, self.y_train)
        self.train_data = gluon.data.DataLoader(dataset_train, self.batch_size, shuffle=True, last_batch='keep')
        
        X_test = Normolise_nd(nd.concat(data_test, noise[-self.test_size:], dim=0), self.num_channel)
        dataset_test = gluon.data.ArrayDataset(X_test, self.y_test)
        self.test_data = gluon.data.DataLoader(dataset_test, self.batch_size, shuffle=True, last_batch='keep')        

    
    def check_acc(self, data_iterator):
        numerator = 0.
        denominator = 0.
        for batch_i, (data, label) in enumerate(data_iterator):
            _, output = self.loss(data, label, train = False)
            predictions = nd.argmax(output, axis=1).as_in_context(ctx)
            numerator += nd.sum(predictions == label.as_in_context(ctx))
            denominator += data.shape[0]
            print('Evaluating accuracy. (complete percent: {:.2f}/100)'.format(1.0 * batch_i / (self.train_size/self.batch_size) * 100 /2)+' '*20, end='')
            sys.stdout.write("\r")        

        return (numerator / denominator).asscalar()


    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        
        checkpoint = {
          'model': self.model,
#           'update_rule': self.update_rule,
          'lr_decay': self.lr_decay,
#           'optim_config': self.optim_config,
          'batch_size': self.batch_size,
#           'num_train_samples': self.num_train_samples,
#           'num_val_samples': self.num_val_samples,
          'epoch': self.epoch,
          'loss_history': self.loss_history,
          'loss_v_history': self.loss_v_history,
          'moving_loss_history': self.moving_loss_history,
          'train_acc_history': self.train_acc_history,
          'test_acc_history': self.test_acc_history,
        }
        
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        nd.save(filename, checkpoint)