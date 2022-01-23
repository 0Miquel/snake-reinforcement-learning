# Snake Reinforcement Learning

Requirements:
- Python 3.5+
- OpenAI Gym
- Numpy
- Pytorch (cpu)
- PyQT 5 for graphics
- Snake environment: https://github.com/telmo-correa/gym-snake

In this project two different methods have been used to build a reinforcement learning agent capable of playing snake game, achieving better mean score than a human player.
- Tabular method
- Linear neural network

Additionally, a convolutional neural network has been built to achieve better results, but we have not achieved the expected results. We think it is due to lack of hardware resources, as we are unable to train with bigger batch sizes.

Both, linear and convolutional neural network models, can be found in the QNN.py file in NN folder.

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
