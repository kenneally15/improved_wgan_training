Improved Training of Wasserstein GANs
=====================================

Code for reproducing experiments in ["Improved Training of Wasserstein GANs"](https://arxiv.org/abs/1704.00028).


## Prerequisites

- Python, NumPy, TensorFlow, SciPy, Matplotlib
- A recent NVIDIA GPU

## Models

Configuration for all models is specified in a list of constants at the top of
the file. Two models should work "out of the box":

- `python gan_toy.py`: Toy datasets (8 Gaussians, 25 Gaussians, Swiss Roll). 
- `python gan_mnist.py`: MNIST

For the other models, edit the file to specify the path to the dataset in
`DATA_DIR` before running. Each model's dataset is publicly available; the
download URL is in the file.

- `python gan_64x64.py`: 64x64 architectures (this code trains on ImageNet instead of LSUN bedrooms in the paper)
- `python gan_language.py`: Character-level language model
- `python gan_cifar.py`: CIFAR-10

## Improved Metric

Algorithm 1 Battacharyya-GAN, our proposed algorithm. 
Require: alpha, the learning rate. c, the clipping parameter. m, the batch size.  n, the number of iterations of the critic per generator iteration. 
Require:  w, the initial critic parameter. theta, the initial generator’s parameters. 
  1: while theta has not converged do
  2: 	for t = 0, …,n  do
  3:		Sample a batch from the real data
  4:		Sample a batch of prior samples
  5:		Calculate Battacharrya Coefficient   
  6:		if BC <= 0.5 do
  7: 			Wasserstein Gradient
  8: 		else do
  9:			KL-Divergence Gradient
 10:		end if 
 11:		Update w with gradient descent
 12:		Clip w to be Lipschitz
 13:	end for
 14: 	Sample a batch from the real data 
 15: 	Sample a batch of prior samples
 16: 	Calculate Battacharrya Coefficient 
 17:	if BC <= 0.5 do
 18: 		Wasserstein Generator Gradient
 19: 	else do
 20: 		 KL-Divergence Generator Gradient
 21: 	Update theta with gradient descent
 22:  end while





