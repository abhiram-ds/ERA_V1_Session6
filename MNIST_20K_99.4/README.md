# MNIST Architecture to achieve 99.4% Accuracy

### Requirement:
#### Build a MNIST digit identifier such that it achieves the following:
1. 99.4% validation accuracy.   
2. Less than 20k Parameters.   
3. Use all of the concepts covered.   
4. Less than 20 Epochs.  
5. Have used BN, Dropout, a Fully connected layer, have used GAP.  

### Approaches

## Approach: With Avg pooling before fully connected layer

[MNIST_20K_Params_ERA_V1_Session_6.ipynb](https://github.com/abhiram-ds/ERA_V1_Session6/blob/main/MNIST_20K_99.4/MNIST_20K_Params_ERA_V1_Session_6.ipynb)

### Architecture

![image](https://user-images.githubusercontent.com/31658286/120032858-70d9c480-c018-11eb-9ba7-5570d4fdfac3.png)


### Model Summary

![image](https://user-images.githubusercontent.com/31658286/120030151-b5636100-c014-11eb-9d2f-57c161ce4185.png)

Convolution layers: 3 blocks, with each block having 2 Conv Layers of 8, 16 and 32 kernels of size 3x3, stride=1 and padding=1  
Batch Normalization Used  
Dropout used: 10% at the end each of the 3 blocks  
Global Average Pooling used to bring down the channel size from 7x7x32 to 1x1x32  
Total Parameters: 18,962  

### Training Logs

```
Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.1392 Batch_id=468 Accuracy=77.82: 100%|██████████| 469/469 [00:34<00:00, 13.70it/s]
Test set: Average loss: 0.0831, Accuracy: 9762/10000 (97.62%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.1619 Batch_id=468 Accuracy=96.28: 100%|██████████| 469/469 [00:25<00:00, 18.43it/s]
Test set: Average loss: 0.0488, Accuracy: 9853/10000 (98.53%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.1054 Batch_id=468 Accuracy=97.22: 100%|██████████| 469/469 [00:28<00:00, 16.24it/s]
Test set: Average loss: 0.0413, Accuracy: 9882/10000 (98.82%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.0160 Batch_id=468 Accuracy=97.61: 100%|██████████| 469/469 [00:26<00:00, 17.80it/s]
Test set: Average loss: 0.0316, Accuracy: 9895/10000 (98.95%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.0479 Batch_id=468 Accuracy=97.88: 100%|██████████| 469/469 [00:25<00:00, 18.32it/s]
Test set: Average loss: 0.0258, Accuracy: 9916/10000 (99.16%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=0.0774 Batch_id=468 Accuracy=98.05: 100%|██████████| 469/469 [00:25<00:00, 18.47it/s]
Test set: Average loss: 0.0273, Accuracy: 9913/10000 (99.13%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: Loss=0.0824 Batch_id=468 Accuracy=98.22: 100%|██████████| 469/469 [00:25<00:00, 18.47it/s]
Test set: Average loss: 0.0254, Accuracy: 9919/10000 (99.19%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 8
Train: Loss=0.0231 Batch_id=468 Accuracy=98.27: 100%|██████████| 469/469 [00:25<00:00, 18.62it/s]
Test set: Average loss: 0.0239, Accuracy: 9921/10000 (99.21%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 9
Train: Loss=0.0335 Batch_id=468 Accuracy=98.42: 100%|██████████| 469/469 [00:25<00:00, 18.61it/s]
Test set: Average loss: 0.0240, Accuracy: 9927/10000 (99.27%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 10
Train: Loss=0.0880 Batch_id=468 Accuracy=98.38: 100%|██████████| 469/469 [00:25<00:00, 18.11it/s]
Test set: Average loss: 0.0233, Accuracy: 9928/10000 (99.28%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 11
Train: Loss=0.0799 Batch_id=468 Accuracy=98.47: 100%|██████████| 469/469 [00:25<00:00, 18.71it/s]
Test set: Average loss: 0.0221, Accuracy: 9937/10000 (99.37%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 12
Train: Loss=0.0957 Batch_id=468 Accuracy=98.52: 100%|██████████| 469/469 [00:25<00:00, 18.58it/s]
Test set: Average loss: 0.0222, Accuracy: 9935/10000 (99.35%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 13
Train: Loss=0.0260 Batch_id=468 Accuracy=98.61: 100%|██████████| 469/469 [00:25<00:00, 18.63it/s]
Test set: Average loss: 0.0213, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 14
Train: Loss=0.1180 Batch_id=468 Accuracy=98.61: 100%|██████████| 469/469 [00:24<00:00, 18.87it/s]
Test set: Average loss: 0.0218, Accuracy: 9928/10000 (99.28%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 15
Train: Loss=0.0541 Batch_id=468 Accuracy=98.70: 100%|██████████| 469/469 [00:24<00:00, 18.90it/s]
Test set: Average loss: 0.0210, Accuracy: 9937/10000 (99.37%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 16
Train: Loss=0.0576 Batch_id=468 Accuracy=98.96: 100%|██████████| 469/469 [00:25<00:00, 18.66it/s]
Test set: Average loss: 0.0183, Accuracy: 9941/10000 (99.41%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 17
Train: Loss=0.0104 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:25<00:00, 18.55it/s]
Test set: Average loss: 0.0179, Accuracy: 9940/10000 (99.40%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 18
Train: Loss=0.0855 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:25<00:00, 18.52it/s]
Test set: Average loss: 0.0180, Accuracy: 9945/10000 (99.45%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 19
Train: Loss=0.0161 Batch_id=468 Accuracy=99.02: 100%|██████████| 469/469 [00:25<00:00, 18.67it/s]
Test set: Average loss: 0.0180, Accuracy: 9940/10000 (99.40%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.0341 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:25<00:00, 18.59it/s]
Test set: Average loss: 0.0172, Accuracy: 9945/10000 (99.45%)

Adjusting learning rate of group 0 to 1.0000e-03.

```

### Losses and Accuracy

![image](https://github.com/abhiram-ds/ERA_V1_Session6/assets/71654199/c297a0e9-492c-4787-bec8-716d86f79ef1)





