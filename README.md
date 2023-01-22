# FreeAdversarialTraining
This repo contains the PyTorch implementation of the paper 'Adversarial Training for Free!'

In this work, we train and evaluate the [ResNet-50](https://arxiv.org/abs/1512.03385) model on the Intel Image Classification dataset, which can be found [here](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

In order to run the notebook locally, make sure to meet the following library requirements:
- tqdm
- gdown
- torch

Talking about our work, we first train our model through a simple PyTorch training loop and afterwards using the Free Adversarial training algorithm proposed in the paper, shown below:<br>

===================================================================<br>
**Algorithm** Free Adversarial Training (Free-*m*)<br>
**Require:** Training samples *X*, perturbation bound $\epsilon$, learning rate $\tau$, hop steps *m* 

1: Initialize $\theta$ <br>
2: $\delta \gets 0$  <br>
3: for epoch= 1 ... N$_{ep}$/*m* do <br>
4: &nbsp;&nbsp;&nbsp;&nbsp;for minibatch *B* $\subset$ *X* do <br>
5: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for i = 1 ... *m* do <br>
6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update $\theta$ with stochastic gradient descent <br>
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$g_\theta$ $\gets$ $\mathbb{E}$ $(x,y) \in B$ [$\nabla_\theta$ $l(x+\delta,y,\theta)]$ <br>
8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$g_{adv}$ $\gets$ $\nabla$ $xl(x+\delta,y,\theta)]$ <br>
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\theta \gets \theta - \tau g_\theta$ <br>
10: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Use gradients calculated for the minimization step to update $\delta$ <br>
11: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\delta \gets \delta + \epsilon \cdot$ sign($g_{adv}$) <br>
12: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\delta \gets$ clip($\delta$, $-\epsilon$, $\epsilon$) <br>
13: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;end for <br>
14: &nbsp;&nbsp;&nbsp;&nbsp;end for <br>
15: end for

==================================================================

The main point and goal of this algorithm is building a model which is robust to PGD-attacks, but that at the same time is cheap and fast to train (7 to 30 times faster than other strong adversarial training methods).

Later on, we validate both models using the PyTorch's common validation loop and finally on PGD-20/-40/-100 attacks. 

Results, as requested, can be seen at the very bottom of the .ipynb notebook file.

In order to quickly validate the best models without having to re-run the training loop, we do provide two pre-trained models in order to replicate the final results:
- ResNet50 with Intel Image Classification trained with Free Adversarial Training *m = 2* : [Link](https://drive.google.com/file/d/1U_dBfCp8P-DmLnvQ1qtuT84bVRoXgBf8/view?usp=share_link) (*~180 MB*)
- ResNet50 with Intel Image Classification trained with Free Adversarial Training *m = 3* : [Link](https://drive.google.com/file/d/1rT0O5Qz0ABDMCd3GeTrxm3S6kEs0sifz/view?usp=share_link) (*~180 MB*)
- ResNet50 with Intel Image Classification trained with Free Adversarial Training *m = 5* : [Link](https://drive.google.com/file/d/13Lv0Yc_O9YX5c54EqekXZvgldJ63s_W4/view?usp=share_link) (*~180 MB*)
- ResNet50 with Intel Image Classification trained with standard PyTorch Training: [Link](https://drive.google.com/file/d/1u9xHf9lPuTOJdANUE0q9aDPY9zNi0jYd/view?usp=share_link) (*~180 MB*)

For testing, simply go to the "Validation (for testing purposes)" section, load the model downloaded earlier and run the validation cell. 
