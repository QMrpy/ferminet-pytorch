# Implementation of FermiNet in PyTorch

This is a pure PyTorch implementation DeepMind's Ferminet (https://arxiv.org/pdf/1909.02487.pdf). 

Till now, the kinetic energy of an electron given its position and wavefunction has been implemented and is working correctly. The implementation deviates from the original paper, in that it directly calculates the kinetic energy from the Laplacian, and not by first finding it from the logarithm of the wavefunction.

Tests for the correctness of the kinetic energy operator has been implemented, using the wavefunction for an electron in a 3D box.

The implementation will try to follow the original implementation from DeepMind (https://github.com/deepmind/ferminet) wherever appropriate.