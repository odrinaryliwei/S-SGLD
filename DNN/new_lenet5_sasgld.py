#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 19:19:19 2021

@author: OrzinaryW
"""
import tensorflow as tf
import scipy as sp
import numpy as np
from tensorflow import keras
import matplotlib
from matplotlib import pyplot as plt
from scipy.special import expit, logit
tf.keras.backend.set_floatx('float64')
class_names = ['Tshirt/top','Trouser','Pull','Dress','Coat','Sandal',
               'Shirt','Sneaker','Bag','Ankle boot']
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

#preprocess data
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')
img_rows, img_cols = 28, 28
X_train = X_train/255.0
X_train = X_train - np.mean(X_train, axis = 0)
X_test = X_test/255.0
X_test = X_test - np.mean(X_test, axis = 0)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)
y_train = tf.convert_to_tensor(y_train)
y_test = tf.convert_to_tensor(y_test)
input_shape = (img_rows, img_cols, 1)

#network structure and prior parameters
num_classes = 10
num_features = img_rows*img_cols
numIter = 10000
input_shape = [img_rows,img_cols,1]
num_layers = np.array([300,200,num_classes])
num_channel = np.array([6,16])
kernel_size = np.array([3,3])
sple_size = len(y_train)
rho_0 = 10000
rho_1 = 1
u_0 = 50
flatten = 784
gamma = np.array(2e-6)
minibatch_size = 100
sigmasq = 1.0
conv_layer = 2
pool_layer = 2
full_layer = 3
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(num_channel[0], (3, 3), activation='relu',input_shape= input_shape ,padding = 'same',strides = 1))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(num_channel[1], (3, 3), activation='relu', padding = 'same',strides = 1))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(num_layers[0],activation = "relu"))
model.add(keras.layers.Dense(num_layers[1],activation = "relu"))
model.add(keras.layers.Dense(num_layers[2],activation = 'softmax'))

p = 0
pre_channel = 1
for i in range(len(num_channel)):
    p += pre_channel*kernel_size[i]**2*num_channel[i] + num_channel[i]
    pre_channel = num_channel[i]
pre_feature = num_features
for  i in range(len(num_layers)):
    p += (pre_feature+1)*num_layers[i]
    pre_feature = num_layers[i]

const_q = (u_0+1)*np.log(p) + 0.5*np.log(rho_0/rho_1)
scale = 1.0*sple_size
#p,const_q,scale
sparse_size = np.array([10,1,10,1,300,10,200,10,10,0])
conv_layer_idx = np.array([0,2])

#Initiate param structure
conv_param_name = ['ConvW1','Convb1','ConvW2','Convb2']
full_param_name = ['W1','b1','W2','b2','W3','b3']

param,param_struct,param_struct_tmp = {},{},{}
for i in range(conv_layer):
    if i == 0:
        param_struct[conv_param_name[2*i]] = np.ones(shape = (kernel_size[i],kernel_size[i],1,num_channel[i]))
        param_struct_tmp[conv_param_name[2*i]] = np.ones(shape = (kernel_size[i],kernel_size[i],1,num_channel[i]))
    else:
        param_struct[conv_param_name[2*i]] = np.ones(shape = (kernel_size[i],kernel_size[i],num_channel[i-1],num_channel[i]))
        param_struct_tmp[conv_param_name[2*i]] = np.ones(shape = (kernel_size[i],kernel_size[i],num_channel[i-1],num_channel[i]))
    param[conv_param_name[2*i]] = model.trainable_variables[2*i].numpy()
    param[conv_param_name[2*i+1]] = model.trainable_variables[2*i+1].numpy()
    param_struct[conv_param_name[2*i+1]] = np.ones(shape = num_channel[i])
    param_struct_tmp[conv_param_name[2*i+1]] = np.ones(shape = num_channel[i])

for i in range(full_layer):
    if i == 0:
        param_struct[full_param_name[2*i]] = np.ones(shape = (flatten,num_layers[i]))
        param_struct_tmp[full_param_name[2*i]] = np.ones(shape = (flatten,num_layers[i]))
    else:
        param_struct[full_param_name[2*i]] = np.ones(shape = (num_layers[i-1],num_layers[i]))
        param_struct_tmp[full_param_name[2*i]] = np.ones(shape = (num_layers[i-1],num_layers[i]))
    param[full_param_name[2*i]] = model.trainable_variables[2*(conv_layer+i)].numpy() 
    param[full_param_name[2*i+1]] = model.trainable_variables[2*(conv_layer+i)+1].numpy() 
    param_struct[full_param_name[2*i+1]] = np.ones(shape = num_layers[i])
    param_struct_tmp[full_param_name[2*i+1]] = np.ones(shape = num_layers[i])
        
coordset = {}

#sparsify and grad functions
def sparsify(model, param, param_struct):
    for i in range(conv_layer):
        weights, biases = param[conv_param_name[2*i]], param[conv_param_name[2*i+1]]
        weights = weights*param_struct[conv_param_name[2*i]]
        biases = biases*param_struct[conv_param_name[2*i+1]]
        model.layers[conv_layer_idx[i]].set_weights([weights, biases])
    for i in range(full_layer):
        weights, biases = param[full_param_name[2*i]], param[full_param_name[2*i+1]]
        weights = weights*param_struct[full_param_name[2*i]]
        biases = biases*param_struct[full_param_name[2*i+1]]
        model.layers[i+conv_layer+pool_layer+1].set_weights([weights, biases])#plus 1 is the flatten layer
    return model

def grad_loss(model, X, y):
    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:  
        y_ = model(X)
        lp_value = loss_fun(y, y_)
    return tape.gradient(lp_value, model.variables)

#updateSparsityStructure
def updateSparsityStructure(param_struct, param_struct_tmp, coordset):
    for i in range(conv_layer):
        if i == 0:
            sel_W = np.random.choice(a = kernel_size[i]*kernel_size[i]*num_channel[i],size = sparse_size[2*i],replace = False)
            sel_b = np.random.choice(a = num_channel[i],size = sparse_size[2*i+1],replace = False)
        else:
            sel_W = np.random.choice(a = kernel_size[i]*kernel_size[i]*num_channel[i-1]*num_channel[i],size = sparse_size[2*i],replace = False)
            sel_b = np.random.choice(a = num_channel[i],size = sparse_size[2*i+1],replace = False)
        param_struct_tmp[conv_param_name[2*i]][:] = param_struct[conv_param_name[2*i]][:]
        param_struct_tmp[conv_param_name[2*i+1]][:] = param_struct[conv_param_name[2*i+1]][:]
        np.reshape(param_struct_tmp[conv_param_name[2*i]],-1)[sel_W] = 0
        param_struct_tmp[conv_param_name[2*i+1]][sel_b] = 0
        coordset[conv_param_name[2*i]] = sel_W
        coordset[conv_param_name[2*i+1]] =sel_b
        
    for i in range(full_layer):
        if i == 0:
            sel_W = np.random.choice(a=num_features * num_layers[i],size = sparse_size[(conv_layer+i)*2], replace =False)
            sel_b = np.random.choice(a=num_layers[i],size = sparse_size[(conv_layer+i)*2+1], replace =False)
        else:
            sel_W = np.random.choice(a = num_layers[i-1] * num_layers[i],size = sparse_size[(conv_layer+i)*2],replace = False)
            sel_b = np.random.choice(a = num_layers[i],size = sparse_size[(conv_layer+i)*2+1],replace = False)
        param_struct_tmp[full_param_name[2*i]][:] = param_struct[full_param_name[2*i]][:]
        param_struct_tmp[full_param_name[2*i+1]][:] = param_struct[full_param_name[2*i+1]][:]
        np.reshape(param_struct_tmp[full_param_name[2*i]],-1)[sel_W] = 0
        param_struct_tmp[full_param_name[2*i+1]][sel_b] = 0
        coordset[full_param_name[2*i]] = sel_W
        coordset[full_param_name[2*i+1]] =sel_b 
    return param_struct_tmp, coordset

#update param Structure
def updateStructure(model,param,param_struct,param_struct_tmp,coordset,minibatch_X,minibatch_y):
    model = sparsify(model, param, param_struct_tmp)
    gradients = grad_loss(model, minibatch_X, minibatch_y)
    #update structure of delta
    for i in range(conv_layer):
        weights, biases = param[conv_param_name[2*i]], param[conv_param_name[2*i+1]]
        weight_index = coordset[conv_param_name[2*i]]
        bias_index = coordset[conv_param_name[2*i+1]]
        #update W
        grad = gradients[2*i].numpy()
        grad = np.reshape(grad,-1)[weight_index]
        vec_temp = scale*grad
        weights = np.reshape(weights,-1)[weight_index]
        zz1 = -weights* vec_temp
        zz2 = 0.5*(rho_1-rho_0)*(weights**2)
        zz = const_q - zz1 + zz2 - 0.5*(zz1**2)
        prob = expit(-zz)
        np.reshape(param_struct[conv_param_name[2*i]],-1)[weight_index] = np.random.binomial(1,prob)
        #update b
        grad = gradients[2*i+1].numpy()
        biases = biases[bias_index]
        vec_temp = scale*grad[bias_index]
        zz1 = -biases* vec_temp
        zz2 = 0.5*(rho_1-rho_0)*(biases**2)
        zz = const_q - zz1 + zz2 - 0.5*(zz1**2)
        prob = expit(-zz)
        param_struct[conv_param_name[2*i+1]][bias_index] = np.random.binomial(1,prob)
        
    for i in range(full_layer):
        weights, biases = param[full_param_name[2*i]], param[full_param_name[2*i+1]]
        weight_index = coordset[full_param_name[2*i]]
        bias_index = coordset[full_param_name[2*i+1]]
        #update W
        grad = gradients[2*(conv_layer+i)].numpy()
        grad = np.reshape(grad,-1)[weight_index]
        vec_temp = scale*grad
        weights = np.reshape(weights,-1)[weight_index]
        zz1 = -weights* vec_temp
        zz2 = 0.5*(rho_1-rho_0)*(weights**2)
        zz = const_q - zz1 + zz2 - 0.5*(zz1**2)
        prob = expit(-zz)
        np.reshape(param_struct[full_param_name[2*i]],-1)[weight_index] = np.random.binomial(1,prob)
        #update b
        grad = gradients[2*(conv_layer+i)+1].numpy()
        vec_temp = scale*grad[bias_index]
        biases = biases[bias_index]
        zz1 = -biases* vec_temp
        zz2 = 0.5*(rho_1-rho_0)*(biases**2)
        zz = const_q - zz1 + zz2 - 0.5*(zz1**2)
        prob = expit(-zz)
        param_struct[full_param_name[2*i+1]][bias_index] = np.random.binomial(1,prob)
    return param,param_struct,model

#update Parameter
def updateParam(model, param,param_struct,minibatch_X,minibatch_y):
    model = sparsify(model, param, param_struct)
    gradients = grad_loss(model, minibatch_X, minibatch_y)
    #Updatae conv layer:
    for i in range(conv_layer):
        L = np.sum(param_struct[conv_param_name[2*i]]==0)
        if L > 0:
            param[conv_param_name[2*i]][param_struct[conv_param_name[2*i]]==0] = np.random.normal(0.0, np.sqrt(1/rho_0), L)
        L = np.sum(param_struct[conv_param_name[2*i+1]]==0)
        if L > 0:
            param[conv_param_name[2*i+1]][param_struct[conv_param_name[2*i+1]]==0] = np.random.normal(0.0, np.sqrt(1/rho_0), L)    
        L = np.sum(param_struct[conv_param_name[2*i]]==1)
        if L > 0:
            sub_grad = gradients[2*i][param_struct[conv_param_name[2*i]]==1].numpy()
            sub_grad = -scale*sub_grad - rho_1*param[conv_param_name[2*i]][param_struct[conv_param_name[2*i]]==1]
            sub_weights = param[conv_param_name[2*i]][param_struct[conv_param_name[2*i]]==1] + 0.5*gamma*sub_grad + np.sqrt(gamma)*np.random.normal(0.0, 1.0,L)
            param[conv_param_name[2*i]][param_struct[conv_param_name[2*i]]==1] = sub_weights
        L = np.sum(param_struct[conv_param_name[2*i+1]]==1)
        if L > 0:
            sub_grad = gradients[2*i+1][param_struct[conv_param_name[2*i+1]]==1].numpy()
            sub_grad = -scale*sub_grad - rho_1*param[conv_param_name[2*i+1]][param_struct[conv_param_name[2*i+1]]==1]
            sub_bias = param[conv_param_name[2*i+1]][param_struct[conv_param_name[2*i+1]]==1] + 0.5*gamma*sub_grad + np.sqrt(gamma)*np.random.normal(0.0, 1.0,L)
            param[conv_param_name[2*i+1]][param_struct[conv_param_name[2*i+1]]==1] = sub_bias
            
    for i in range(full_layer):
        L = np.sum(param_struct[full_param_name[2*i]]==0)
        if L > 0:
            param[full_param_name[2*i]][param_struct[full_param_name[2*i]]==0] = np.random.normal(0.0, np.sqrt(1/rho_0), L)
        L = np.sum(param_struct[full_param_name[2*i+1]]==0)
        if L > 0:
            param[full_param_name[2*i+1]][param_struct[full_param_name[2*i+1]]==0] = np.random.normal(0.0, np.sqrt(1/rho_0), L)    
        L = np.sum(param_struct[full_param_name[2*i]]==1)
        if L > 0:
            sub_grad = gradients[2*(conv_layer+i)][param_struct[full_param_name[2*i]]==1].numpy()
            sub_grad = -scale*sub_grad - rho_1*param[full_param_name[2*i]][param_struct[full_param_name[2*i]]==1]
            sub_weights = param[full_param_name[2*i]][param_struct[full_param_name[2*i]]==1] + 0.5*gamma*sub_grad + np.sqrt(gamma)*np.random.normal(0.0, 1.0,L)
            param[full_param_name[2*i]][param_struct[full_param_name[2*i]]==1] = sub_weights
        L = np.sum(param_struct[full_param_name[2*i+1]]==1)
        if L > 0:
            sub_grad = gradients[2*(conv_layer+i)+1][param_struct[full_param_name[2*i+1]]==1].numpy()
            sub_grad = -scale*sub_grad - rho_1*param[full_param_name[2*i+1]][param_struct[full_param_name[2*i+1]]==1]
            sub_bias = param[full_param_name[2*i+1]][param_struct[full_param_name[2*i+1]]==1] + 0.5*gamma*sub_grad + np.sqrt(gamma)*np.random.normal(0.0, 1.0,L)
            param[full_param_name[2*i+1]][param_struct[full_param_name[2*i+1]]==1] = sub_bias
    return param, model

#main function
Output = np.zeros(shape=[numIter,2])
count = 0
for kk in range(numIter):
    #Draw a mini batch
    rand_index = np.random.choice(sple_size, minibatch_size)
    minibatch_X, minibatch_y = tf.gather(X_train,rand_index), tf.gather(y_train,rand_index)
    
    # Update sparsity structure
    param_struct_tmp, coordset = updateSparsityStructure(param_struct, param_struct_tmp, coordset)
    
    #update structure of delta
    param, param_struct,model = updateStructure(model,param,param_struct,param_struct_tmp,coordset,minibatch_X,minibatch_y)

    #update parameter
    param, model = updateParam(model,param,param_struct,minibatch_X,minibatch_y)
    
    if (kk+1)%100== 0:
        model = sparsify(model, param, param_struct)
        y_pred = model(X_test)
        pred = np.argmax(y_pred, axis=-1)
        Output[count,0] = np.sum(pred == y_test) / len(y_test)
        nnz = 0
        for i in range(conv_layer):
            nnz += np.count_nonzero(param_struct[conv_param_name[2*i]]) + np.count_nonzero(param_struct[conv_param_name[2*i+1]]) 
        for i in range(full_layer):
            nnz += np.count_nonzero(param_struct[full_param_name[2*i]]) + np.count_nonzero(param_struct[full_param_name[2*i+1]]) 
        Output[count,1] = nnz/p
        print([kk, Output[count,:]])
        count+=1

#plot results
line1, = plt.plot(Output[:count,0])
line2, = plt.plot(Output[:count,1])
plt.xlabel('Iterations')
plt.ylabel('Percentage')
plt.legend([line1, line2], ['Prediction Accuracy', 'Sparsity'], loc = 'best')
