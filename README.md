# Snake Reinforcement Learning

The objective of this project is to compare different algorithms and approaches to play snake game using reinforcement learning.

Two algorithms have been tested:
- Q-learning
- SARSA

And two different approaches:
- Tabular method
- Linear neural network

Additionally, a convolutional neural network has been built to achieve better results, but we have not achieved the expected results. We think it is due to lack of hardware resources, as we are unable to train with bigger batch sizes.

Both, linear and convolutional neural network models, can be found in the QNN.py file in NN folder.

## Requirements
- Python 3.5+
- OpenAI Gym
- Numpy
- Pytorch (cpu)
- PyQT 5 for graphics
- Snake environment: https://github.com/telmo-correa/gym-snake

## Train and test
### Neural Network
Neural network training, the model is saved in the models folder.
```
git clone https://github.com/0Miquel/snake-reinforcement-learning.git
cd source/NN
python TrainNN.py
```
Neural network testing, it executes the previously trained model.
```
python TestNN.py
```
