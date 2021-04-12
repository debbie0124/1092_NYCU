# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 02:19:04 2021

@author: User
"""

import argparse
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt


### class of FNN model
class FNN:
    def __init__(self, configs, weight_dict, train_image, train_label, test_image, test_label, imgfile=None, weight_scale=0.1):
        self.configs = configs
        self.weight_dict = weight_dict
        self.train_image = train_image
        self.train_label = train_label
        self.test_image = test_image
        self.test_label = test_label
        self.train_label_onehot = np.eye(6)[self.train_label]    # one-hot encoder
        self.test_label_onehot = np.eye(6)[self.test_label]    # one-hot encoder
        
        for layer in self.weight_dict.keys():
            self.weight_dict[layer] = self.weight_dict[layer] * weight_scale
        
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []
        self.z_dict = {}
        self.act_dict = {}
        self.delta_dict = {}
        
    
    # def img_preprocess()
    
    def activation(self, data, mode):
        if mode == 'relu':
            return np.maximum(data, 0)
        elif mode == 'softmax':
            return np.apply_along_axis(lambda x: np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x))), 1, data)
    
    
    def cross_entropy(self, pred, true):
        return -np.mean(np.multiply(true, np.log(pred+1e-15)))
    
    
    def calculus(self, layer, act, label):
        if act == 'relu':
            relu_prime = lambda x: 1 if x>0 else 0
            relu_deriv = np.vectorize(relu_prime)(self.z_dict[layer])
            self.delta_dict[layer] = np.multiply(relu_deriv, np.dot(self.delta_dict[layer+1], self.weight_dict[layer+1].T))
        elif act == 'softmax':
            self.delta_dict[layer] = self.act_dict[layer] - label
        dw = np.dot(self.act_dict[layer-1].T, self.delta_dict[layer])
        return dw
    
    
    def cal_loss_acc(self, data, label_onehot, label):
        pred_onehot = data
        for layer, cfg in self.configs['nn'].items():
            pred_onehot = self.activation(np.nan_to_num(np.dot(pred_onehot, self.weight_dict[layer])), cfg['act'])
        it_loss = self.cross_entropy(pred_onehot, label_onehot)
        it_pred = np.argmax(pred_onehot, axis=1)
        it_accuracy = np.sum(it_pred == label)/len(data)
        return it_loss, it_accuracy
    
    
    def backpropagation(self):
        iteration = int(np.ceil(len(self.train_image) / self.configs['batch_size']))
        # test_batch_size = int(np.ceil(len(self.test_image) / iteration))
        
        ### epoch
        for ep in range(self.configs['epoch']):
            print('epoch: ', ep)
            accuracyList_train = []
            lossList_train = []
            accuracyList_test = []
            lossList_test = []
            
            ### iteration
            for it in range(iteration):
                print('iteration: ', it)
                train_it = self.train_image[it*self.configs['batch_size']: it*self.configs['batch_size']+self.configs['batch_size']]
                train_it_label_onehot = self.train_label_onehot[it*self.configs['batch_size']: it*self.configs['batch_size']+self.configs['batch_size']]
                train_it_label = self.train_label[it*self.configs['batch_size']: it*self.configs['batch_size']+self.configs['batch_size']]
                # test_it = self.test_image[it*test_batch_size: it*test_batch_size+test_batch_size]    # 配合 train 的 iteration 次數
                # test_it_label_onehot = self.test_label_onehot[it*test_batch_size: it*test_batch_size+test_batch_size]
                # test_it_label = self.test_label[it*test_batch_size: it*test_batch_size+test_batch_size]
                
                ### forwardpass
                self.act_dict[-1] = train_it
                for layer, cfg in self.configs['nn'].items():
                    self.z_dict[layer] = np.nan_to_num(np.dot(self.act_dict[layer-1], self.weight_dict[layer]))   # 避免計算出現 NaN 問題
                    self.act_dict[layer] = self.activation(self.z_dict[layer], cfg['act'])
                    
                ### get loss
                # loss = cross_entropy(act_dict[layer], train_it_label)
                
                ### backwardpass
                for layer, cfg in reversed(self.configs['nn'].items()):
                    dw = self.calculus(layer, cfg['act'], train_it_label_onehot)
                    self.weight_dict[layer] = self.weight_dict[layer] - self.configs['lr'] / len(train_it) * dw
                
                ### accracy and loss of train
                train_it_loss, train_it_accuracy = self.cal_loss_acc(train_it, train_it_label_onehot, train_it_label)
                lossList_train.append(train_it_loss)
                accuracyList_train.append(train_it_accuracy)
                
                ### accracy and loss of test
                test_loss, test_accuracy = self.cal_loss_acc(self.test_image, self.test_label_onehot, self.test_label)
                lossList_test.append(test_loss)
                accuracyList_test.append(test_accuracy)
            
            self.train_accuracy.append(accuracyList_train)
            self.train_loss.append(lossList_train)
            self.test_accuracy.append(accuracyList_test)
            self.test_loss.append(lossList_test)
        return self.train_accuracy, self.train_loss, self.test_accuracy, self.test_loss
    
    
    def plot_figure(self, train, test, title):
        plt.plot(np.array(train).flatten(), color='green', label='Training '+title, alpha=0.5)
        plt.plot(np.array(test).flatten(), color='orange', label='Test '+title)
        plt.xlabel('Iteration (based on batch={})'.format(self.configs['batch_size']))
        if title == 'ACC':
            plt.ylabel('Accuracy %')
        else:
            plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.show()
    
    
    def output_txt(self, data):
        for layer, cfg in self.configs['nn'].items():
            data = self.activation(np.nan_to_num(np.dot(data, self.weight_dict[layer])), cfg['act'])
        output = np.argmax(data, axis=1)
        np.savetxt('output.txt', output, newline='', fmt='%d')
    

### main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='given the config of FNN model')
    parser.add_argument('--weight', help='given the weight of FNN model')
    parser.add_argument('--imgfilelistname', help='given the test data')
    args = parser.parse_args()
    
    if args.config:
        with open(args.config) as jsonfile:
            configs = json.load(jsonfile)
        for i, layer in enumerate(configs['nn']):
            configs['nn'][i] = configs['nn'].pop(layer)
    else:    # 未指定參數時預設
        configs = {'nn': {0: {'input_dim': 1024, 
                              'output_dim': 2048, 
                              'act': 'relu'},
                          1: {'input_dim': 2048, 
                              'output_dim': 512, 
                              'act': 'relu'},
                          2: {'input_dim': 512, 
                              'output_dim': 6, 
                              'act': 'softmax'}},
                   'epoch': 20,
                   'lr': 0.001,
                   'batch_size': 2048,
                   'criterion': 'cross_entropy'}
    
    if args.weight:
        with np.load(args.weight) as weightfile:
            weight_dict = dict(weightfile)
        for i, layer in enumerate(weight_dict):
            weight_dict[i] = weight_dict.pop(layer)
    else:    # 未指定參數時預設
        weight_dict = {}
        for layer, cfg in configs['nn'].items():
            weight_dict[layer] = np.random.normal(0.0, 1.0, size=(cfg['input_dim'], cfg['output_dim']))
        
    if args.imgfilelistname:
        # myArr = np.loadtxt('my_txt.txt', delimiter=',')
        with open(args.imgfilelistname, encoding='utf-8') as imglistfile:
            imgList = imglistfile.read().splitlines()
            imgfilelistname = np.array([np.asarray(Image.open(img).convert('L')) / 255 for img in imgList])
        imgfilelistname = np.reshape(imgfilelistname, (imgfilelistname.shape[0], imgfilelistname.shape[1]*imgfilelistname.shape[2]))
    else:    # 未指定參數時預設
        imgfilelistname = None
        
    
    ### 讀檔(train, test)
    with np.load('./train.npz') as train:
        train_image = train['image'] / 255
        train_label = train['label']
    with np.load('./test.npz') as test:
        test_image = test['image'] / 255
        test_label = test['label']
    
    ### train, test 前處理
    train_image = np.reshape(train_image, (train_image.shape[0], train_image.shape[1]*train_image.shape[2]))
    test_image = np.reshape(test_image, (test_image.shape[0], test_image.shape[1]*test_image.shape[2]))
    
    ### 若沒有參數 imgfilelistname，則執行第一小題；若有，則執行第二小題
    if type(imgfilelistname) == type(None):
        
        ### Homework 1-1 1(a) FNN model
        # print('Homework 1-1 1(a)')
        # FNN_model_1a = FNN(configs, weight_dict, train_image, train_label, test_image, test_label)
        # FNN1a_train_accuracy, FNN1a_train_loss, FNN1a_test_accuracy, FNN1a_test_loss = FNN_model_1a.backpropagation()
        # FNN_model_1a.plot_figure(FNN1a_train_accuracy, FNN1a_test_accuracy, 'ACC')
        # FNN_model_1a.plot_figure(FNN1a_train_loss, FNN1a_test_loss, 'Loss')
        
        ### Homework 1-1 1(b) turn the batch_size
        # print('Homework 1-1 1(b)')
        # for batch in [512, 12288, 25500]:#[2048, 8192, 12750]:
        #     configs['batch_size'] = batch
        #     for layer, cfg in configs['nn'].items():
        #         weight_dict[layer] = np.random.normal(0.0, 1.0, size=(cfg['input_dim'], cfg['output_dim']))
        #     FNN_model_1b = FNN(configs, weight_dict, train_image, train_label, test_image, test_label)
        #     FNN1b_train_accuracy, FNN1b_train_loss, FNN1b_test_accuracy, FNN1b_test_loss = FNN_model_1b.backpropagation()
        #     FNN_model_1b.plot_figure(FNN1b_train_accuracy, FNN1b_test_accuracy, 'ACC')
        #     FNN_model_1b.plot_figure(FNN1b_train_loss, FNN1b_test_loss, 'Loss')
        # configs['batch_size'] = 2048    # 還原參數
        
        # # ### Homework 1-1 1(c) turn the weight
        print('Homework 1-1 1(c)')
        for layer, cfg in configs['nn'].items():
            weight_dict[layer] = np.random.normal(0.0, 1.0, size=(cfg['input_dim'], cfg['output_dim']))
        weight_zero = {}
        for layer, cfg in configs['nn'].items():
            weight_zero[layer] = np.zeros([cfg['input_dim'], cfg['output_dim']])
        #for weight in [weight_dict, weight_zero]:
        for weight in [weight_dict, weight_zero]:
            FNN_model_1c = FNN(configs, weight, train_image, train_label, test_image, test_label)
            FNN1c_train_accuracy, FNN1c_train_loss, FNN1c_test_accuracy, FNN1c_test_loss = FNN_model_1c.backpropagation()
            FNN_model_1c.plot_figure(FNN1c_train_accuracy, FNN1c_test_accuracy, 'ACC')
            FNN_model_1c.plot_figure(FNN1c_train_loss, FNN1c_test_loss, 'Loss')
    
    # else:
        ### Homework 1-2 given parameters
        # print('Homework 1-1 2(a)')
        # FNN_model_2 = FNN(configs, weight_dict, train_image, train_label, test_image, test_label, imgfile=imgfilelistname)
        # FNN2_train_accuracy, FNN2_train_loss, FNN2_test_accuracy, FNN2_test_loss = FNN_model_2.backpropagation()
        # FNN_model_2.output_txt(imgfilelistname)



