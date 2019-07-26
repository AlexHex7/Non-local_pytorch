# Non-local_pytorch
- Implementation of [**Non-local Neural Block**](https://arxiv.org/abs/1711.07971).

## Statement
- You can find different kinds of non-local block in **lib/**. 
- The code is tested on MNIST dataset. You can select the type of non-local block 
in **lib/network.py**.
- If there is something wrong in my code, please contact me, thanks!


## Environment
- python 3.6 or python 3.7.3
- pytorch 0.4.1 or pytorch 1.1.0


## Update Records
1. Figure out how to implement the **concatenation** type, and add the code to **lib/**.
2. Fix the bug in **lib/non_local.py** (old version) when using multi-gpu. Someone shares the 
reason with me, and you can find it in [here](https://github.com/pytorch/pytorch/issues/8637).
3. Fix the error of 3D pooling in **lib/non_local.py** (old version). Appreciate 
[**protein27**](https://github.com/AlexHex7/Non-local_pytorch/issues/17) for pointing it out.
4. For convenience, I split the **lib/non_local.py** into four python files, and move the 
old versions (**lib/non_loca.py** and **lib/non_local_simple_version.py**) into 
**lib/backup/**.
5. Modify the code to support pytorch 0.4.1, and move the code supporting pytorch 0.3.1 \
to **Non-Local_pytorch_0.3.1/**.
6. Test the code with pytorch 1.1.0 and it works.

## Running Steps
1. Select the type of non-local block in **lib/network.py**.
    ```
    from lib.non_local_concatenation import NONLocalBlock2D
    from lib.non_local_gaussian import NONLocalBlock2D
    from lib.non_local_embedded_gaussian import NONLocalBlock2D
    from lib.non_local_dot_product import NONLocalBlock2D
2. Run **demo_MNIST.py** with one GPU or multi GPU.
    ```
    CUDA_VISIBLE_DEVICES=0,1 python demo_MNIST.py

## Todo
- Experiments on Charades dataset.
- Experiments on COCO dataset.


## Related Repositories
1. [**Non-local ResNet-50 TSM**](https://github.com/MIT-HAN-LAB/temporal-shift-module) 
([**Paper**](https://arxiv.org/abs/1811.08383)) on Kinetics dataset. They report that their model achieves a good performance 
of **75.6% on Kinetics**, which is even higher than Non-local ResNet-50 I3D 
([**Here**](https://github.com/AlexHex7/Non-local_pytorch/issues/23)).