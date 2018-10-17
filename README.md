# non-local_pytorch
- Implementation of [**Non-local Neural Networks**](https://arxiv.org/abs/1711.07971).

## Statement
- Only do the experiments on MNIST dataset so far.
- You can find the non-local block in **lib/**. 
- The code can support **multi-gpu** now.
- If there is something wrong in my code, please contact me, thanks!

There are two version **non-local.py** and **non-local-simple-version.py**. 

- **non-local.py** contains the implementation of Gaussian, embedded Gaussian and  dot product, which is mainly for learning.
- **non-local-simple-version.py** only contains  the implementation of embedded Gaussian.

## Environment
- python 3.6
- pytorch 0.3.0

## Todo
- Experiments on Kinetics dataset.
- Experiments on Charades dataset.
- Experiments on COCO dataset.
- [x] Make sure how to do the Implementation of concatenation.
- [x] Support multi-gpu.
- [x] Fix the bug in **lib/non_local.py** when using multi-gpu (thanks for someone share the reason (you can find in [here](https://github.com/pytorch/pytorch/issues/8637))).