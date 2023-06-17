# MNIST Architecture to achieve 99.4% Accuracy

### Requirement:
#### Build a MNIST digit identifier such that it achieves the following:
1. 99.4% validation accuracy.   
2. Less than 20k Parameters.   
3. Use all of the concepts covered.   
4. Less than 20 Epochs.  
5. Have used BN, Dropout, a Fully connected layer, have used GAP.  

### Approaches

We have adopted two approaches to achieve the target of 99.4 % accuracy. One using Average pooling and one with Max pooling, before the fully connected layer. The reason for talking about the one with Max pooling is because it has provided us with the best Accuracy and least loss. **Approach 1 is our submission, as per the requirements.**

## Approach 1: With Avg pooling before fully connected layer

[MNIST_20K_Params_EVA6_Session_4.ipynb](https://github.com/RohinSequeira/EVA6_Session4_Backpropagation_and_Architectural_Basics/blob/main/MNIST_with_99.4/MNIST_20K_Params_EVA6_Session_4.ipynb)

### Architecture

![image](https://user-images.githubusercontent.com/31658286/120032858-70d9c480-c018-11eb-9ba7-5570d4fdfac3.png)


### Model Summary

![image](https://user-images.githubusercontent.com/31658286/120030151-b5636100-c014-11eb-9d2f-57c161ce4185.png)

Convolution layers: 3 blocks, with each block having 2 Conv Layers of 8, 16 and 32 kernels of size 3x3, stride=1 and padding=1  
Batch Normalization Used  
Dropout used: 10% at the end each of the 3 blocks  
Global Average Pooling used to bring down the channel size from 7x7x32 to 1x1x32  
Total Parameters: 18,962  

![image](https://user-images.githubusercontent.com/31658286/120031587-9e257300-c016-11eb-9242-11a36f3cbc66.png)


### Training Logs

```
Epoch 1 : 
Train set: Average loss: 0.2658, Accuracy: 81.20

Test set: Average loss: 0.110, Accuracy: 96.65

Epoch 2 : 
Train set: Average loss: 0.0619, Accuracy: 97.35

Test set: Average loss: 0.041, Accuracy: 98.66

Epoch 3 : 
Train set: Average loss: 0.0540, Accuracy: 98.05

Test set: Average loss: 0.039, Accuracy: 98.71

Epoch 4 : 
Train set: Average loss: 0.0376, Accuracy: 98.38

Test set: Average loss: 0.037, Accuracy: 98.85

Epoch 5 : 
Train set: Average loss: 0.0350, Accuracy: 98.64

Test set: Average loss: 0.028, Accuracy: 99.04

Epoch 6 : 
Train set: Average loss: 0.0745, Accuracy: 98.72

Test set: Average loss: 0.026, Accuracy: 99.14

Epoch 7 : 
Train set: Average loss: 0.0307, Accuracy: 98.84

Test set: Average loss: 0.025, Accuracy: 99.18

Epoch 8 : 
Train set: Average loss: 0.0736, Accuracy: 98.92

Test set: Average loss: 0.024, Accuracy: 99.27

Epoch 9 : 
Train set: Average loss: 0.0105, Accuracy: 98.98

Test set: Average loss: 0.022, Accuracy: 99.31

Epoch 10 : 
Train set: Average loss: 0.0064, Accuracy: 99.05

Test set: Average loss: 0.020, Accuracy: 99.39

Epoch 11 : 
Train set: Average loss: 0.0446, Accuracy: 99.15

Test set: Average loss: 0.023, Accuracy: 99.27

Epoch 12 : 
Train set: Average loss: 0.0319, Accuracy: 99.13

Test set: Average loss: 0.020, Accuracy: 99.27

Epoch 13 : 
Train set: Average loss: 0.0253, Accuracy: 99.18

Test set: Average loss: 0.022, Accuracy: 99.23

Epoch 14 : 
Train set: Average loss: 0.0097, Accuracy: 99.28

Test set: Average loss: 0.020, Accuracy: 99.39

Epoch 15 : 
Train set: Average loss: 0.0032, Accuracy: 99.18

Test set: Average loss: 0.021, Accuracy: 99.35

Epoch 16 : 
Train set: Average loss: 0.0061, Accuracy: 99.28

Test set: Average loss: 0.017, Accuracy: 99.44

Epoch 17 : 
Train set: Average loss: 0.0091, Accuracy: 99.25

Test set: Average loss: 0.020, Accuracy: 99.25

Epoch 18 : 
Train set: Average loss: 0.0069, Accuracy: 99.28

Test set: Average loss: 0.018, Accuracy: 99.42

Epoch 19 : 
Train set: Average loss: 0.0429, Accuracy: 99.38

Test set: Average loss: 0.019, Accuracy: 99.40

Epoch 20 : 
Train set: Average loss: 0.0026, Accuracy: 99.41

Test set: Average loss: 0.018, Accuracy: 99.44
```

### Losses and Accuracy

![image](https://user-images.githubusercontent.com/31658286/120030553-3fabc500-c015-11eb-89bf-e4101541c18d.png)

![image](https://user-images.githubusercontent.com/31658286/120030649-5c47fd00-c015-11eb-86e1-d87f55ecbe3a.png)



## Approach 2: With Max pooling before fully connected layer

[MNIST_digit_recognition_Session_4.ipynb](https://github.com/RohinSequeira/EVA6_Session4_Backpropagation_and_Architectural_Basics/blob/main/MNIST_with_99.4/MNIST_digit_recognition_Session_4.ipynb)

### Architecture

![image](https://user-images.githubusercontent.com/31658286/120028032-c65ea300-c011-11eb-8774-54f017fa96eb.png)


### Model Summary

![image](https://user-images.githubusercontent.com/31658286/120031110-f90a9a80-c015-11eb-9a97-080afababd40.png)

Convolution layers: 4 layers of 3x3  
Batch Normalization Used  
Dropout used: 3%  
Total Parameters: 16890  


![image](https://user-images.githubusercontent.com/31658286/120031195-1dff0d80-c016-11eb-98d6-a08a1db5fd05.png)



### Training Logs

```
Epoch 1 : 
Train set: Average loss: 0.1048, Accuracy: 95.09

Test set: Average loss: 0.055, Accuracy: 98.52

Epoch 2 : 
Train set: Average loss: 0.0179, Accuracy: 98.60

Test set: Average loss: 0.035, Accuracy: 99.04

Epoch 3 : 
Train set: Average loss: 0.0167, Accuracy: 98.87

Test set: Average loss: 0.029, Accuracy: 99.09

Epoch 4 : 
Train set: Average loss: 0.0023, Accuracy: 99.04

Test set: Average loss: 0.028, Accuracy: 99.24

Epoch 5 : 
Train set: Average loss: 0.0290, Accuracy: 99.20

Test set: Average loss: 0.032, Accuracy: 98.97

Epoch 6 : 
Train set: Average loss: 0.0244, Accuracy: 99.31

Test set: Average loss: 0.025, Accuracy: 99.19

Epoch 7 : 
Train set: Average loss: 0.0169, Accuracy: 99.28

Test set: Average loss: 0.026, Accuracy: 99.17

Epoch 8 : 
Train set: Average loss: 0.0040, Accuracy: 99.70

Test set: Average loss: 0.019, Accuracy: 99.41

Epoch 9 : 
Train set: Average loss: 0.0122, Accuracy: 99.75

Test set: Average loss: 0.018, Accuracy: 99.45

Epoch 10 : 
Train set: Average loss: 0.0022, Accuracy: 99.81

Test set: Average loss: 0.018, Accuracy: 99.40

Epoch 11 : 
Train set: Average loss: 0.0019, Accuracy: 99.81

Test set: Average loss: 0.017, Accuracy: 99.49

Epoch 12 : 
Train set: Average loss: 0.0051, Accuracy: 99.88

Test set: Average loss: 0.018, Accuracy: 99.44

Epoch 13 : 
Train set: Average loss: 0.0014, Accuracy: 99.85

Test set: Average loss: 0.018, Accuracy: 99.42

Epoch 14 : 
Train set: Average loss: 0.0054, Accuracy: 99.86

Test set: Average loss: 0.017, Accuracy: 99.48

Epoch 15 : 
Train set: Average loss: 0.0002, Accuracy: 99.91

Test set: Average loss: 0.017, Accuracy: 99.45

Epoch 16 : 
Train set: Average loss: 0.0013, Accuracy: 99.92

Test set: Average loss: 0.017, Accuracy: 99.46

Epoch 17 : 
Train set: Average loss: 0.0016, Accuracy: 99.94

Test set: Average loss: 0.017, Accuracy: 99.47

Epoch 18 : 
Train set: Average loss: 0.0014, Accuracy: 99.93

Test set: Average loss: 0.017, Accuracy: 99.48

Epoch 19 : 
Train set: Average loss: 0.0074, Accuracy: 99.92

Test set: Average loss: 0.017, Accuracy: 99.45

Epoch 20 : 
Train set: Average loss: 0.0344, Accuracy: 99.92

Test set: Average loss: 0.017, Accuracy: 99.47
```

### Losses and Accuracy

![image](https://user-images.githubusercontent.com/31658286/120031330-4d157f00-c016-11eb-81ce-b401fe73b1fb.png)


![image](https://user-images.githubusercontent.com/31658286/120031381-5bfc3180-c016-11eb-8cd0-744e130d8767.png)
