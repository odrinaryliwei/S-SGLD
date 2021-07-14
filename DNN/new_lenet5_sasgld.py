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

sparse_size = np.array([10,1,10,1,300,10,200,10,10,0])
conv_layer_idx = np.array([0,2])

p = 0
LL = len(model.variables) 
for i in range(LL):
    p += np.prod(model.variables[i].shape)

const_q = (u_0+1)*np.log(p) + 0.5*np.log(rho_0/rho_1)
scale = 1.0*sple_size
p,const_q,scale

#Initiate param structure
param,param_struct,param_struct_tmp = {},{},{}
for i in range(LL):
        param[i] = model.variables[i].numpy()
        param_struct[i] = np.ones(shape = model.variables[i].shape)
        param_struct_tmp[i] = np.ones(shape = model.variables[i].shape)
coordset = {}

#sparsify and grad functions
def sparsify(model, param, param_struct):
    for i in range(conv_layer):
        weights = param[2*i]*param_struct[2*i]
        bias = param[2*i+1]*param_struct[2*i+1]
        model.layers[conv_layer_idx[i]].set_weights([weights, bias])
    for i in range(full_layer):
        weights = param[2*(conv_layer+i)]*param_struct[2*(conv_layer+i)]
        bias = param[2*(conv_layer+i)+1]*param_struct[2*(conv_layer+i)+1]
        model.layers[conv_layer+pool_layer+1+i].set_weights([weights, bias])
    return model

def grad_loss(model, X, y):
    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:  
        y_ = model(X)
        lp_value = loss_fun(y, y_)
    return tape.gradient(lp_value, model.variables)

#updateSparsityStructure
def updateSparsityStructure(param_struct, param_struct_tmp, coordset):
    for i in range(LL):
        sel_W = np.random.choice(a = np.prod(model.variables[i].shape),size = sparse_size[i],replace = False)
        param_struct_tmp[i][:] = param_struct[i][:]
        np.reshape(param_struct_tmp[i],-1)[sel_W] = 0
        coordset[i] = sel_W
    return param_struct_tmp, coordset

#update param Structure
def updateStructure(model,param,param_struct,param_struct_tmp,coordset,minibatch_X,minibatch_y):
    model = sparsify(model, param, param_struct_tmp)
    gradients = grad_loss(model, minibatch_X, minibatch_y)
    #update structure of delta
    for i in range(LL):
        weights, weight_index = param[i],coordset[i]
        #update W
        grad = gradients[i].numpy()
        grad = np.reshape(grad,-1)[weight_index]
        vec_temp = scale*grad
        weights = np.reshape(weights,-1)[weight_index]
        zz1 = -weights* vec_temp
        zz2 = 0.5*(rho_1-rho_0)*(weights**2)
        zz = const_q - zz1 + zz2 - 0.5*(zz1**2)
        prob = expit(-zz)
        np.reshape(param_struct[i],-1)[weight_index] = np.random.binomial(1,prob)
    return param,param_struct,model

#update Parameter
def updateParam(model, param,param_struct,minibatch_X,minibatch_y):
    model = sparsify(model, param, param_struct)
    gradients = grad_loss(model, minibatch_X, minibatch_y)
    #Updatae conv layer:
    for i in range(LL):
        L = np.sum(param_struct[i]==0)
        if L > 0:
            param[i][param_struct[i]==0] = np.random.normal(0.0, np.sqrt(1/rho_0), L)
        L = np.sum(param_struct[i]==1)
        if L > 0:
            sub_grad = gradients[i][param_struct[i]==1].numpy()
            sub_grad = -scale*sub_grad - rho_1*param[i][param_struct[i]==1]
            sub_weights = param[i][param_struct[i]==1] + 0.5*gamma*sub_grad + np.sqrt(gamma)*np.random.normal(0.0, 1.0,L)
            param[i][param_struct[i]==1] = sub_weights
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
        for i in range(LL):
            nnz += np.count_nonzero(param_struct[i])
        Output[count,1] = nnz/p
        print(count,Output[count,:])
        count+=1
    

#plot results
line1, = plt.plot(Output[:count,0])
line2, = plt.plot(Output[:count,1])
plt.xlabel('Iterations')
plt.ylabel('Percentage')
plt.legend([line1, line2], ['Prediction Accuracy', 'Sparsity'], loc = 'best')
