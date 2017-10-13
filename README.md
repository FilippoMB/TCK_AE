### Deep Kernelized Auto Encoder with Time series Cluster Kernel 
#### Learning compressed representations of blood samples time series with missing data
------

Tensorflow implementation of a [Deep Kernelized Auto Encoder (dkAE)](https://link.springer.com/chapter/10.1007/978-3-319-59126-1_35), 
aligned with the [Time series Cluster Kernel (TCK)](https://arxiv.org/abs/1704.00794), 
for learning vectorial representations of mutlivariate time series (MTS) with missing data.


The MTS analyzed are blood samples measurements of patient with surgical site infections.
The original dataset is available [at this link](https://groups.google.com/forum/#!topic/ml-news/MQtVkxizrrU).
Here, we consider a subset of 883 MTS divided in 2 classes: light infections and severe infections.
Each MTS has 10 attributes recorded for 20 time steps.

------
