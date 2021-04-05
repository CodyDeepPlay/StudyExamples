# This document records some of the useful commands


## Python


#### GPU setup

* First import your tensorflow in python, and test whether CUDA supports Tensorflow

```python
# importing the tensorflow package
import tensorflow as tf
tf.test.is_built_with_cuda()
```
This command should return 'True' or 'False' in the console. 'True' means CUDA supports for your Tensorflow installation. 


* Then, make sure GPU is available for Tensorflow to use.

Next, we want to confirm that the GPU is available to Tensorflow, use a built-in utility function in Tensorflow:
```python
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
```
This command will return 'True' or 'False'. 'True' means the GPU is available for Tensorflow to use. 

* For a newer version of TF, the command to check GPU is available is different.

You might need to use the following command if your Tensorflow is with a newer version:
```python
tf.config.list_physical_devices('GPU')
```

In Anaconda promt, type the following command
'''Anaconda
conda list cudnn
'''
This should return the information "Name Version Build Channel". If this is not empty, means Anaconda has install cudnn, which might conflict with installed version on windows. 
