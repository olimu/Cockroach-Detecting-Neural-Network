# Cockroach Detecting Neural Network
Neural Network trained to detect cockroches and generate heat maps highlighting key features used to detect cockroaches, namely shape. 

code folder
- LibModel1L.py is a module which includes support code to train, run, evaluate, and debug the network
   - shrink322.tgz is my image dataset.
- The NN_and_Heat_Maps.ipynb jupyter notebook first trains the neural network, runs it, and outputs its metric results. Then it generates heat maps for specific images.
  - The heat map code outputs an array of heat maps for each layer. Currently there are 5 image paths at the begining of this code, 1 verification image, 1 true negative, 1 false hit, 1 true positive, and 1 miss. 



**CockroachNNPoster.pdf:** poster from Pittsburgh Regional Science and Engineering Fair

**CockroachNNSlides.pdf:** slides that goes with audio from the Pennsylvania Junior Academy Science (PJAS) competition

**CockroachNNAudio.mp4:** audio that goes with slides from PJAS competition

**Appendices.pdf:** contains data related to the neural network, specifically many heat maps that illustrate key features
