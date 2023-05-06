# EE399-A-HW4

## Neural Networks
Marcus Chen, May 5, 2023

Neural networks are the current standard for machine learning, and as an extension, artificial intelligence as a whole. Instead of using single models to do classification and prediction, neural networks provide the opportunity to evaluate data that is processed across multiple models or “layers.” In this project, we will create feed forward neural networks for the linear data from Project 1 and the MNIST data from Project 3 and compare them to their more traditional machine learning counterparts. 

### Introduction:
In previous projects, we analyzed increasingly specific models in use for predictive data optimization and classification, but since 2014, neural networks have become the main standard for performing the same tasks. Instead of just using singular models in order to perform data prediction, neural networks incorporate multiple models, or “layers” in order to create a set of predictive data and classifications. 

In the first part of this project, the linear dataset from Project 1 is used: 
```
X = np.arange(0,31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```

Instead of using a sinusoidal or polynomial model as done previously, a three-layer feed forward neural network (FFNN) is made customized to the data. 

Similar to the aforementioned project, the first 20 data points are used as the training data to create a predictive model of 11 other points. The first and last 10 data points are used as the next set of training data for a predictive model. The results of all of these will be compared to the accuracy of the singular model predictions. 

In the second part of this project, the MNIST dataset from Project 3 is used:

The data is first computed for the first 20 PCA modes, and then an FFNN is created customized to the data to classify the digits. The results of the FFNN are then compared to similar results using a long short-term memory neural network (LSTM) and the models used in Project 3: the SVM and the decision tree. 

### Theoretical Background:
#### Parts of the Neural Network:
<img width="710" alt="Screenshot 2023-05-05 184420" src="https://user-images.githubusercontent.com/66970342/236591701-617dddcf-7186-4991-89be-21aa8826bb2d.png">


##### Input Layer: 
The layer that receives input and transfers them to the succeeding layers in the network; the number of neurons is the same as the number of features/attributes in the dataset
##### Hidden Layer: 
The layers that impose transformations between the input and output layers
##### Neurons: 
An adaptation of biological neurons, they determine the sum of weighted inputs and initiate an activation function to normalize the sum
##### Activation Functions: 
The functions that create linear or nonlinear transformations for the data being processed. It is very important to pick activation functions that are easy to differentiate.

Examples of Activation Functions:
###### Linear:
$f(x)=ax+b$

###### Binary Step:
$f(x)=u(x-n)$

###### Sigmoid:
$f(x)=\frac{1}{1+e^{-x}}$

###### ReLU:
$f(x)=xu(x-n)$

##### Weights: 
The strength or magnitude of connection between two neurons, similar to coefficients in standard linear regression.

##### Output Layer: 
The forecasted feature

#### Feed Forward Neural Network (FFNN): 
The simplest form of a neural network: the connection between nodes or neurons does not form a cycle, therefore making information move only one direction, forwards.

##### Back Propagation: 
A computation of the gradient of a loss function, with respect to the weights of a network, iterating backwards through the layers to avoid redundant calculation.

Back propagation adjusts and updates the weights by this formula:
$\vec{W_{k+1}}=\vec{W_k}-\delta \nabla E$

where $\delta$ refers to the learning rate and $\nabla E$ is the gradient descent, or the iterative optimization algorithm to find a local minimum.

#### Long Short-term Memory Neural Network (LSTM): 
A recurrent neural network (RNN) with feedback connections

### Algorithm Interpretation and Development:
In order to create the FFNN’s for our data, we use pytorch. The design of the network itself is done like so:
```
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
To process the data, the data is converted into something that can be usable for pytorch like so:
```
X_tensor = torch.tensor(X)
Y_tensor = torch.tensor(Y)
```
To create the training and testing model, we can choose the indexes we want:
```
dataset_train_1_X = torch.unsqueeze(X_tensor[0:20], 1)
dataset_train_1_Y = torch.unsqueeze(Y_tensor[0:20], 1)
dataset_test_1_X = torch.unsqueeze(X_tensor[20:31], 1)
dataset_test_1_Y = torch.unsqueeze(Y_tensor[20:31], 1)
```
For when the training data is non-continuous, we can use torch.cat:
```
dataset_train_1_X = torch.unsqueeze(torch.cat((X_tensor[0:10], X_tensor[21:31])), 1)
dataset_train_1_Y = torch.unsqueeze(torch.cat((Y_tensor[0:10], Y_tensor[21:31])), 1)
```
To train the data, the programming is done like so:
```
net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(num_epochs):
      x = dataset_train_1_X.to(dtype=torch.float32)
      y = dataset_train_1_Y.to(dtype=torch.float32)
      optimizer.zero_grad()
      outputs = net(x)
      loss = criterion(outputs, y)
      loss.backward()
      optimizer.step()
        

      print ('Epch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, 1, len(dataset_train_1_Y), loss.item()))
```
In regards to the loss function, we could either base it on the MSE:
```
criterion = nn.MSELoss()
```
or on classification:
```
criterion = nn.CrossEntropyLoss()
```
To test the data, the programming is done like so:
```
with torch.no_grad():
    total = 0.0
    x = dataset_test_1_X.float()
    y = dataset_test_1_Y.float()
    outputs = net(x)
    MSE = (outputs - y)**2
    total += MSE
        
    print('MSE Error: {}'.format(torch.mean(total)))
```
In order to create and use a LSTM model, the programming is done like so:
```
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        pass
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.hidden_size)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out)
        return out
       
        pass
pass
model = RNN(20, 128, 2, 10)
```

### Computational Results:
Based on a FFNN model going over 100 epochs using the first 20 points as training data the results are,
```
Epch [100/100], Step [1/20], Loss: 5.4793
MSE Error: 5.165465354919434
```
If we refer to the MSE error values from project 1
```
Linear Model: 2.242749387090776, 3.363619366080294
Parabolic Model: 2.1255393483520155, 8.71366660302094
19th degree polynomial model: 0.028351481277572182, 28626352734.19632
```
the training mean square error is worse than all the ones using traditional models. The FFNN has a larger error than the linear model when predicting the testing data by a factor of 1.5, but beats the parabolic and polynomial model by over 3 and millions respectively, and remains consistent with the training value.

Based on a FFNN model going over 100 epochs using the first 10 and last 10 points as training data the results are, 
```
Epch [100/100], Step [1/20], Loss: 3.3812
MSE Error: 7.677873134613037
```
If we refer to the MSE error values from project 1
```
Linear Model: 1.8516699046029184, 2.73091076355018
Parabolic Model: 1.8508364117779978, 2.7052339602955877
19th degree polynomial model: 532.5689882067522, 463.31672158927245
```
the training mean square error is better than the polynomial by a factor of over 157, but worse than the linear and parabolic models by a factor of 2. The FFNN has a larger error on the test data than the linear and the parabolic model by a factor of almost 3, but is still much better than the polynomial model by a factor of 60. 

By using a FFNN to predict classifications for the MNIST dataset, after 10 epochs these are the results:
```
Accuracy of the network on the 10000 test images: 93.53 %
```
After 100 epochs these are the results:
```
Accuracy of the network on the 10000 test images: 96.98 %
```

By using an LSTM to predict classifications of the MNIST dataset, after 10 epochs these are the results:
```
Accuracy of the network on the 10000 test images: 61.29 %
```
After 100 epochs these are the results:
```
Accuracy of the network on the 10000 test images: 96.47 %
```

In comparison to the LSTM, the FFNN initially performs much better than the LSTM, but after 100 epochs, they approach a similar level of accuracy.

In comparison to the SVM and decision tree classifier
```
SVM Accuracy: 97.35000000000001 %
Decision Tree Accuracy: 79.08 %
```

The accuracy of the SVM is still greater than either neural network, but if we were to add more epochs, the data could approach that level of accuracy. Despite that, the FFNN at 10 epochs and 100 epochs and the LSTM at 100 epochs performs better than the decision tree overall.

### Conclusions:
In comparison to standard linear models, neural networks use a combination of comparatively simpler models in order to predict results for a variety of tasks, from linear regression, to classification. Even though the simple FFNN's we used for linear regression and classification were not initially as accurate as the singular models, adding epochs allowed reinforcement for the NN to become increasingly more accurate with its data prediction for both testing and training data. 

Because of the volumes of neurons that exist in a simple neural network, it is also much easier to create networks that overfit to the specific training data, hence the issues shown with the linear regression. Similar to the previous linear models, when working with neural networks, it is imperative to choose epochs and learning rate that specify to the data, but not to the point of overfitting.
