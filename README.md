# non-local_pytorch
- Implementation of [**Non-local Neural Networks**](https://arxiv.org/abs/1711.07971).

## Statement
- Only do the experiments on MNIST dataset so far.
- You can find the non-local block in **lib/**. 
- If there is something wrong in my code, please contact me, thanks!

There are two version **non-local.py** and **non-local-simple-version.py**. 

- **non-local.py** contains the implementation of Gaussian, embedded Gaussian and  dot product,
- **non-local-simple-version.py** only contains  the implementation of embedded Gaussian.

## Environment
- python 3.6
- pytorch 0.3.0

## Todo
- Experiments on Kinetics dataset.
- Experiments on Charades dataset.
- Experiments on COCO dataset.
- Make sure how to do the Implementation of concatenation.
