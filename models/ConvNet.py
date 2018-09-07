#!usr/bin/python
#coding=utf-8

# importing the basic library
from __future__ import print_function

from layers import *
from utils import *
ctx = check_ctx()

# importing MxNet >= 1.0
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import autograd, gluon

import random



def init_params(input_dim, output_dim, conv_params, pool_params, fc_params, weight_scale = .01,):
    #######################
    #  Set the scale for weight initialization (random_normal)
    #######################
    mx.random.seed(1)
    random.seed(1)
    
    filters, kernels, stride, padding, dilate = conv_params['num_filter'], conv_params['kernel'], \
                                                conv_params['stride'], conv_params['padding'], conv_params['dilate']
    kernels_pool, stride_pool, padding_pool, dilate_pool =  pool_params['kernel'], pool_params['stride'], \
                                                            pool_params['padding'], pool_params['dilate']
    hidden_dim = fc_params['hidden_dim']
    
    params = {}
    F_in, H, W = input_dim
    
    # CNN ##########################################################################################################
    for i, (nf, k, S, P, D, k_p, S_p, P_p, D_p) in enumerate(zip(filters, kernels, stride, padding, dilate, 
                                                                 kernels_pool, stride_pool, padding_pool, dilate_pool,)):
        
        params['W{:d}'.format(i+1,)] = nd.random_normal(loc=0, scale=weight_scale, shape=(nf,)+(F_in,)+k, ctx=ctx )
        params['b{:d}'.format(i+1,)] = nd.random_normal(shape=nf, scale=weight_scale, ctx=ctx)
        F_in = nf
        
        # 计算数据流的维度 (可以单独打包为一个公式)
        H = (H - k[0] -(k[0]-1)*(D[0]-1) +2*P[0])//S[0] +1
        W = (W - k[1] -(k[1]-1)*(D[1]-1) +2*P[1])//S[1] +1
        H = (H - k_p[0] -(k_p[0]-1)*(D_p[0]-1) +2*P_p[0])//S_p[0] +1
        W = (W - k_p[1] -(k_p[0]-1)*(D_p[0]-1) +2*P_p[1])//S_p[1] +1
        i_out = i
    # MLP ##########################################################################################################
    hd_in = nf * H * W
    for j, hd in enumerate(hidden_dim):
        
        params['W{:d}'.format(j+i_out+2,)] = nd.random_normal(loc=0, scale=weight_scale, shape=(hd_in, hd), ctx=ctx )
        params['b{:d}'.format(j+i_out+2,)] = nd.random_normal(shape=hd, scale=weight_scale, ctx=ctx)
        hd_in = hd
        j_out = j
        
    # OUTPUT ##########################################################################################################
    params['W{:d}'.format(j_out+i_out+3,)] = nd.random_normal(loc=0, scale=weight_scale, shape=(hd_in, output_dim), ctx=ctx )
    params['b{:d}'.format(j_out+i_out+3,)] = nd.random_normal(shape=output_dim, scale=weight_scale, ctx=ctx)       
    
    return params




class ConvNet(object):
    
    def __init__(self, input_dim = (1, 1, 8192), output_dim = 2,  params_inits = None,
                 conv_params = None, act_params = None, pool_params = None, fc_params = None, dtype = None):
        """
        Initialize a Convolution Network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.dtype = dtype
        self.output_dim = output_dim
        self.input_dim = input_dim
        
        # conv_params = {'kernel': ((1,16), (1,8), (1,8)), 
        #                'num_filter': (16//2, 32//2, 64//2),
        #                'stride': ((1,1), (1,1), (1,1)),
        #                'padding': ((0,0), (0,0), (0,0)),
        #                'dilate': ((1,1), (1,1), (1,1))}
        # act_params = {'act_type': ('relu', 'relu', 'relu', 'relu', 'relu')}
        # pool_params = {'pool_type': ('max', 'max', 'max'),
        #                'kernel': ((1,8), (1,8), (1,8)),
        #                'stride': ((1,2), (1,2), (1,2)),
        #                'padding': ((0,0), (0,0), (0,0)),
        #                'dilate': ((1,1), (1,1), (1,1))}
        # fc_params = {'hidden_dim': (64, 64)}
        
        try: 
            check_dict_dim(conv_params) 
        except: 
            loc = locals()
            print('The dimension of {} is wrong!'.format(get_variable_name(conv_params, loc)))
            raise
        
        try:
            check_dict_dim(pool_params)
        except Exception as e:
            loc = locals()
            print('The dimension of {} is wrong!'.format(get_variable_name(pool_params, loc)))
            raise
        
        # 检查一下参数和超参数的基本维度对应关系

        # CNN超参数 ##########################################################################################################
        if conv_params:
            self.conv_params = conv_params
        else:
            raise NameError("Parameters 'conv_params' are not defined!")
        if act_params:
            self.act_params = act_params
        else:
            raise NameError("Parameters 'act_params' are not defined!")        
        if pool_params:
            print('Loading the params...')
            self.pool_params = pool_params
        else:
            self.pool_params = {'pool_type': ('avg', 'avg', 'avg'),
                                'kernel': ((1,1), (1,1), (1,1)),
                                'stride': ((1,1), (1,1), (1,1)),
                                'padding': ((0,0), (0,0), (0,0))}
        if fc_params:
            self.fc_params = fc_params
        else:
            self.fc_params = None
            raise NameError("Parameters 'fc_params' are not defined!")


        # 模型参数 ##########################################################################################################
        if params_inits:
            self.params = params_inits
        else:
            self.params = init_params(input_dim, output_dim, conv_params, pool_params, fc_params,)        
            
        # 精度 ##########################################################################################################
        # for k, v in self.params.items():
        #     self.params[k] = v.astype(dtype)



    def network(self, X=None, debug=False, drop_prob = 0.25, ):
                
        filters, kernels, stride, padding, dilate = self.conv_params['num_filter'], self.conv_params['kernel'], \
                                                    self.conv_params['stride'], self.conv_params['padding'], self.conv_params['dilate']
        type_pool, kernels_pool, stride_pool, padding_pool, dilate_pool =  self.pool_params['pool_type'], \
                                                                           self.pool_params['kernel'], self.pool_params['stride'], \
                                                                           self.pool_params['padding'], self.pool_params['dilate']
        act_type = self.act_params['act_type']
        hidden_dim = self.fc_params['hidden_dim']
        
        
        # CNN ##########################################################################################################
        convlayer_out = X
        interlayer = []
        for i, (nf, k, S, P, D, t_p, k_p, S_p, P_p, D_p, a) in enumerate(zip(filters, kernels, stride, padding, dilate, 
                                                                     type_pool, kernels_pool, stride_pool, padding_pool, dilate_pool,
                                                                     act_type)):
            W, b = self.params['W{:d}'.format(i+1,)], self.params['b{:d}'.format(i+1,)]
            convlayer_out = nd.Convolution(data = convlayer_out, weight=W, bias=b, kernel=k, num_filter=nf, stride=S, dilate=D)
            convlayer_out = activation(convlayer_out, act_type = a)
            convlayer_out = nd.Pooling(data=convlayer_out, pool_type=t_p, kernel=k_p, stride=S_p, pad=P_p)

            interlayer.append(convlayer_out)
            i_out = i
            if debug:
                print("layer{:d} shape: {}".format(i+1, convlayer_out.shape))
        
        # MLP ##########################################################################################################
        FClayer_out = nd.flatten(convlayer_out)
        interlayer.append(FClayer_out)
        if debug:
            print("After Flattened, Data shape: {}".format(FClayer_out.shape))

        for j, (hd, a) in enumerate(zip(hidden_dim, act_type[-len(hidden_dim):])):
            W, b = self.params['W{:d}'.format(j+i_out+2,)], self.params['b{:d}'.format(j+i_out+2,)]
            FClayer_out = nd.dot(FClayer_out, W) + b
            FClayer_out = activation(FClayer_out, act_type = a)
            
            if autograd.is_training():
                # 对激活函数的输出使用droupout
                FClayer_out = dropout(FClayer_out, drop_prob)
            if debug:
                print("layer{:d} shape: {}".format(j+i_out+2, FClayer_out.shape))
            interlayer.append(FClayer_out)            
            j_out = j
            
        # OUTPUT ##########################################################################################################
        W, b = self.params['W{:d}'.format(j_out+i_out+3,)], self.params['b{:d}'.format(j_out+i_out+3,)]            
        yhat = nd.dot(FClayer_out, W) + b

        if debug:
            print("Output shape: {}".format(yhat.shape))
        interlayer.append(yhat)       

        return yhat, interlayer
    
    
    


if __name__ == '__main__':
    print('CPU or GPU? : ', ctx)
