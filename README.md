# Cockroach-NN
The NN_and_Heat_Maps.ipynb jupyter notebook first trains the neural network, runs it, and outputs its metric results. Then it generates heat maps for specific images.

LibModel1L.py is a module which includes support code to train, run, evaluate, and debug the network. shrink322.tgz is my image dataset.

The heat map code outputs an array of heat maps for each layer. Currently there are 5 image paths at the begining of this code, 1 verification image, 1 true negative, 1 false hit, 1 true positive, and 1 miss. 
