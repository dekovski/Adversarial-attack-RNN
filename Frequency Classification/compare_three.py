# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:51:37 2020

@author: shank
"""


from scipy.optimize import fsolve
import numpy as np
from anytree import Node
import matplotlib.pyplot as plt
import pickle
import sys
import winsound
import theano
from six.moves import cPickle
import theano.tensor as T
import time

jac_x = theano.function(**cPickle.load(open('theano_model/jac_x.pkl', "rb"))) 
jac_h = theano.function(**cPickle.load(open('theano_model/jac_h.pkl', "rb"))) 
gru = theano.function(**cPickle.load(open('theano_model/gru.pkl', "rb"))) 
dy_dh = theano.function(**cPickle.load(open('theano_model/dy_dh.pkl', "rb"))) 
classification = theano.function(**cPickle.load(open('theano_model/y.pkl', "rb"))) 

with open('C:/Users/shank/Documents/Acads/RNNs/Code/RNN_copy_task/GRU_trained.txt', "rb") as fp:
    list_of_params = pickle.load(fp)
h_initial = list_of_params[0]

def B(h_k, x_k, d_k):
    lam = 0
    B = 0
    size = 0.1
    while (lam <= 1) :
        B = B + size*jac_x(np.array([x_k + lam*d_k]), np.array([h_k]))
        lam = lam + size
    return B.reshape(2,1)
    
def fixed_point_disturbance(h_k_nominal,h_k_pert, x_k, e_k, targeted = True):
    d = np.zeros(1)
    scale = 100 #3*1/1.5
    if targeted :
        y_e = np.zeros(2)
        y_e[target_class] = 1
        Mat = dy_dh([h_k_pert]).reshape(2,2)
        e_k = np.matmul(y_e,Mat)
    for i in range(5):
        #d = scale*np.matmul(np.transpose(B(h_k_pert,x_k,d, params)),h_k_pert - h_k_nominal)
        d = eps*np.tanh(scale*np.matmul(e_k,B(h_k_pert,x_k,d)))
    if (np.linalg.norm(d) == 0):
        d = eps*(2*np.random.rand(9)-1)
    return d

def dynamical_fixed_point_disturbance(h_k_nominal,h_k_pert, x_k, d_k):
    d = 0
    scale = 100 #3*1/1.5
    k = 0.8
    y_e = np.zeros(2)
    y_e[target_class] = 1
    Mat = dy_dh([h_k_pert]).reshape(2,2)
    e_k = np.matmul(y_e,Mat)
    d = k*(eps*np.tanh(scale*np.matmul(e_k,B(h_k_pert,x_k,d_k))) - d_k) + d_k
    if (d == 0):
        d = 0.1
    return d
    
def disturbance(h_k,x_k):
    # example : h_k = [1,2], x_k = [1]  
    y_e = np.zeros(2)
    y_e[target_class] = 1
    Mat = dy_dh([h_k]).reshape(2,2)
    e_k = np.matmul(y_e,Mat)
    JAC = jac_x(np.array([x_k]), np.array([h_k])).reshape(2,1)
    d = eps*np.sign(np.matmul(e_k,JAC))
    return d


class_ = 1 #nominal class
target_class = 1 - class_
linear = True
targeted = True
use_quadratic_disturbance = False
multi_iteration = False
disturbance_window = 1
eps = 0.15


X_NOMINAL = np.load('data/X_NOMINAL_class_' + str(class_) + '.npy')

    ### GRADIENT ###
attack_success=0
tic = time.time()
for x_nominal in X_NOMINAL:   
    
    x_nominal = np.array([x_nominal]).T
    h_nominal = np.zeros((101,2))
    y_nominal = np.zeros((100,2))
    h_nominal[0] = h_initial[0]
    
    h_pert_grad = np.zeros((101,2))
    x_pert_grad = np.zeros((100,1))
    y_pert_grad = np.zeros((100,2))
    h_pert_grad[0] = h_initial[0]
    
    for k in range(len(x_nominal)):
        y_nominal[k],h_nominal[k+1] = gru(np.array([x_nominal[k]]),np.array([h_nominal[k]]))
        d_k = disturbance(h_pert_grad[k], x_nominal[k])
        x_pert_grad[k] = x_nominal[k] + d_k
        y_pert_grad[k],h_pert_grad[k+1] = gru(np.array([x_pert_grad[k]]),np.array([h_pert_grad[k]]))
        

    if(np.argmax(classification(np.array([h_pert_grad[-1]]))) != class_):
       attack_success+=1
       
toc = time.time()
result = (toc-tic,attack_success)


np.save('data/compare/grad_class_' + str(class_) + '_' + str(eps) + '.npy', result)


    ### FIXED POINT ###
attack_success=0
tic = time.time()
for x_nominal in X_NOMINAL:        
    
    x_nominal = np.array([x_nominal]).T
    h_nominal = np.zeros((101,2))
    y_nominal = np.zeros((100,2))
    h_nominal[0] = h_initial[0]
    
    h_pert_fp = np.zeros((101,2))
    x_pert_fp = np.zeros((100,1))
    y_pert_fp = np.zeros((100,2))
    h_pert_fp[0] = h_initial[0]
    e = np.zeros((100,2))
    d_k = 0
    
    for k in range(len(x_nominal)):
        y_nominal[k],h_nominal[k+1] = gru(np.array([x_nominal[k]]),np.array([h_nominal[k]]))
        e[k] = h_pert_fp[k] - h_nominal[k]
        d_k = fixed_point_disturbance(h_nominal[k], h_pert_fp[k], x_nominal[k], e[k], True)
        x_pert_fp[k] = x_nominal[k] + d_k
        y_pert_fp[k],h_pert_fp[k+1] = gru(np.array([x_pert_fp[k]]),np.array([h_pert_fp[k]]))
        
    if(np.argmax(classification(np.array([h_pert_fp[-1]]))) != class_):
        attack_success+=1
        
toc = time.time()
result = (toc-tic,attack_success)
np.save('data/compare/fp_class_' + str(class_) + '_' + str(eps) + '.npy', result)

    ### DYNAMIC FIXED POINT ###
attack_success=0
tic = time.time()
for x_nominal in X_NOMINAL:

    x_nominal = np.array([x_nominal]).T
    h_nominal = np.zeros((101,2))
    y_nominal = np.zeros((100,2))
    h_nominal[0] = h_initial[0]
    
    h_pert_fp_dyn = np.zeros((101,2))
    x_pert_fp_dyn = np.zeros((100,1))
    y_pert_fp_dyn = np.zeros((100,2))
    h_pert_fp_dyn[0] = h_initial[0]
    e = np.zeros((100,2))
    d_k_dyn = 0
    
    for k in range(len(x_nominal)):
        y_nominal[k],h_nominal[k+1] = gru(np.array([x_nominal[k]]),np.array([h_nominal[k]]))
        e[k] = h_pert_fp_dyn[k] - h_nominal[k]
        d_k_dyn = dynamical_fixed_point_disturbance(h_nominal[k],h_pert_fp_dyn[k],x_nominal[k], d_k_dyn)
        x_pert_fp_dyn[k] = x_nominal[k] + d_k_dyn
        y_pert_fp_dyn[k],h_pert_fp_dyn[k+1] = gru(np.array([x_pert_fp_dyn[k]]),np.array([h_pert_fp_dyn[k]]))

    if(np.argmax(classification(np.array([h_pert_fp_dyn[-1]]))) != class_):
        attack_success+=1
        
toc = time.time()
result = (toc-tic,attack_success)
np.save('data/compare/fp_dyn_class_' + str(class_) + '_' + str(eps) + '.npy', result)

