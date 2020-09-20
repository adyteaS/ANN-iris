
# coding: utf-8

# #### Loading the iris dataset

# In[1]:


import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# In[2]:


iris = datasets.load_iris()
x = iris.data
y = iris.target

x = (x - np.mean(x,axis = 0))/np.std(x,axis = 0)
#create one encoded target

y_ = np.zeros((y.shape[0],3))
y_[np.arange(y.shape[0]), y] = 1
y = y_


# #### Creating train and test dataset
# 
# randomly permuting the index and taking 135 samples for training and 15 for validation

# In[3]:


def create_train_test_dataset(x,y):
    
    index = np.random.permutation(x.shape[0])
    x = x[index]
    y = y[index]
    train_idx = int(len(index) * 0.9)
    x_tr = x[:train_idx]
    y_tr = y[:train_idx]
    x_te = x[train_idx:]
    y_te = y[train_idx:]
    
    return x_tr, y_tr, x_te, y_te
    



# #### Creating Artificial Neural Network
# 
# 
# 1. First Let's define a list containing the number of hidden units in each layer beginning from input to output, we will use the lenght of the list - 2 to determine number of hidden layers
# 

# In[4]:


# 2 hidden units
l = [4,6,3]


# In[5]:


class ANN():
    
    def __init__(self,l,lr):
        
        self.l = l
        
        self.lr = lr
        
        '''Fully Connected network'''
        
        self.nnet = {}
        
        '''Need input to each layer while performing backpropogation, populated in forward function and used in backward function'''
        
        self.input = {}
        
        '''Delta Weight for weight updation'''
        self.delta_weight = {}
        
        for i in range(len(self.l)-1):
            
            self.nnet[i] = {}
            
            '''Layer weight dimension will be matrix of size n * m, where n = current input dimension and m is number of hidden units in projected layer'''
            
            d_curr = self.l[i]
            
            d_hid = self.l[i+1]
            
            '''Initializing each weight using random standard normal'''
            
            self.nnet[i]['wts'] = np.random.normal(0,1,[d_curr,d_hid])
            
            '''Last layer is bias free'''
            
            if i+1 != len(self.l)-1:
                self.nnet[i]['bias'] = np.random.normal(0,1,[1,d_hid])
            else:
                self.nnet[i]['bias'] = np.zeros([1,d_hid])
                
    
    def sigmoid(self,x):
        
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self,x):
        
        return x * (1-x)
    
    def forward(self,x):
        
        '''For layers in neural network, perform sigmoid(X * W + B)'''
        
        self.input[0] = x
        
        for i in self.nnet.keys():

            '''Numpy dot with 2d matrices performs cross product '''
            x = self.sigmoid(np.dot(x,self.nnet[i]['wts']) + self.nnet[i]['bias'])
            
            '''Saving input to each future layer'''
            
            self.input[i+1] = x 
        
        return x
    
    def update_weights(self):
        
        for key in self.nnet.keys():
            
            self.nnet[key]['wts'] += self.lr * self.delta_weight[key]['wts']
            
            '''Sum biases along rows to meet 1 x n bias dimensions'''
            
            self.nnet[key]['bias'] += self.lr * np.sum(self.delta_weight[key]['bias'],axis = 0,keepdims = True)
    
    def backward(self,loss):
        
      
        
        nl = len(self.nnet.keys())
        
        g = loss * self.sigmoid_derivative(self.input[nl])
        
        self.delta_weight[nl-1] = {}
        
        self.delta_weight[nl-1]['wts'] = np.dot(self.input[nl-1].T,g)
        self.delta_weight[nl-1]['bias'] =  g
        
        for i in range(nl-1,0,-1):
            
            self.delta_weight[i-1] = {}
            
            g = np.dot(g,self.nnet[i]['wts'].T) * self.sigmoid_derivative(self.input[i])
            
            self.delta_weight[i-1]['wts'] = np.dot(self.input[i-1].T,g)
            self.delta_weight[i-1]['bias'] =  g
            
        self.update_weights()
        
        
    def return_nnet(self):
        
        return self.nnet
    
        
nnet = ANN(l,0.001)


# #### Backpropagation
# 
# For the last layer
# 
# 1. Compute loss = y - y_true  
#   {1 x 3} matrix
# 
# 2. gradient = loss .* y .* (1-y)  
#   {1 x 3} matrix
# 
# 3. delta_weight for 6 * 3 matrix = input_to_last_layer.Transpose * gradient {only way your dimensions will match, do this outside the loop}  
#   {1 x 6}.Transpose * {1 x 3} = {6 x 3} weight matrix
# 
# For the hidden layers 2nd last to 2nd
# 
# 1. compute new_gradient = gradient (the above one) * weights_of_next layer (example for 3rd last to 2nd last layer, it will be the weight matrix for neurons, 6 * 3 one as for this example) .* sigmoid_derivative (input_to_6 * 3_layer)  
#   
#   {1 x 3} x {3 x 6} .* {1 x 6} = {1 x 6} matrix
# 
# 2. delta_weight for 6 * 6 matrix = input_to_layer.Transpose * new_gradient  
#   {6 x 1} x {1 x 6} = {6 x 6} matrix 

# In[6]:


it = np.arange(0,135,15)    


# In[7]:


tot_loss = []
val_tot_loss = []

val_class_error = []

trn_class_error = []

for j in range (2000):
    
    x_tr, y_tr, x_te, y_te = create_train_test_dataset(x,y)    
    
    epoch_loss = 0
    
    trn_loss = 0
    
    val_loss = 0
    
    for i in it:
        
        x_batch = x_tr[i:i+15].reshape([-1,4])
        y_batch = y_tr[i:i+15].reshape([-1,3])
        
        y_pred = nnet.forward(x_batch)
    
        # Calculate Loss
        
        dloss = y_batch - y_pred
        
        pred = np.argmax(y_pred,axis = 1)
        true = np.argmax(y_batch,axis = 1)
    
        trn_loss += (1 - np.sum(pred==true)/len(pred)) * 100
                
        nnet.backward(dloss)
        
        epoch_loss += np.mean(np.sum(np.square(dloss)/2,axis = 0))
        
    tot_loss.append(epoch_loss/9)
    
    trn_class_error.append(trn_loss/9)
    
    x_batch = x_te.reshape([-1,4])
    y_batch = y_te.reshape([-1,3])
    
    y_pred = nnet.forward(x_batch)
    
    pred = np.argmax(y_pred,axis = 1)
    true = np.argmax(y_batch,axis = 1)
    
    
    val_class_loss = (1 - np.sum(pred==true)/len(pred)) * 100
    val_class_error.append(val_class_loss)
    
    #pred = np.argmax(y,axis = 0)
        
    dloss = y_batch - y_pred
        
    val_loss = np.mean(np.sum(np.square(dloss)/2,axis = 0))
    
    val_tot_loss.append(val_loss)
        
    #print('Epoch: {}\tLoss: {:.3f} \tVal_Loss: {:.3f}'.format(j, epoch_loss/9, val_loss))
    print('Epoch: {}\tLoss: {:.3f} \tVal_Loss: {:.3f} \tClass_Loss: {:.3f} \tVal_Class_Loss: {:.3f}'.format(j, epoch_loss/9, val_loss,trn_loss/9,val_class_loss))


# In[8]:

#[:200] for printing a graph with range 1:200.
plt.figure(1)
#plt.subplot(211)
plt.plot(tot_loss[:2000],label = 'Training Error')

plt.xlabel('Epochs 1:200')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.figure(2)
#plt.subplot(212)
plt.plot(val_tot_loss[:2000],label = 'Testing Error',color="green")
plt.xlabel('Epochs 1:200')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.figure(3)
#plt.subplot(211)
plt.plot(val_class_error[:2000],label = 'Classification Loss Training')
plt.xlabel('Epochs 1:200')
plt.ylabel('Classification Error')
plt.legend()
plt.figure(4)
#plt.subplot(212)
plt.plot(trn_class_error[:2000],label = 'Classification Loss Testing',color="green")
plt.xlabel('Epochs 1:200')
plt.ylabel('Classification Error')
plt.legend()
plt.show()

