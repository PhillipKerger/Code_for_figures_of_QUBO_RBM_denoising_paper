This folder contains folders of code relating to the QUBO image denoising method for RBMs.
This method is described and analyzed in detail in publication by Phillip Kerger and Ryoji Miyazaki titled
something like "image denoising via QUBO with Boltzmann Machines and Quantum Annealing". 

The code here can be used to reproduce the figures in that paper.  
The bas12x12 and mnist12x12 folders contain code to test the QUBO method against various other methods, 
on a 12x12 pixel bars-and-stripes and a 12x12 pixel MNIST dataset, respectively. In its current form, 
the code also tests the QUBO method with "guessing" sigma and then introducing some bias to the guess for 
a more robust performance (see next paragraph about robustness w.r.t. choice of sigma). 

There is also code in the folder to test the QUBO method against itself with different choices of the parameter \rho, 
via choosing different values of sigma. This reveals that choosing a slightly smaller sigma is better. 
It would be nice to run this on the QA as well, though it may not be absolutely necessary. 
This reveals that practically, choosing 0.75*sigma is a good idea (this makes rho larger, i.e. we become 
slightly less willing to flip pixels -- we will only flip a pixel if our model is *strongly convinced* that 
it should be flipped. This makes the method perform better in practice, which we interpret as a robustness 
modification of the model. 
