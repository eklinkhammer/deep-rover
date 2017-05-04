import numpy as np
import keras.models import Sequential
import keras.layers import Dense, Activation

class FeedForwardNeuralNet(object):
    """ Wrapper around Keras Neural Network for Free Forward Neural Networks.
           Current (and only) use case is for evolutionary algorithms. 
           Therefore, only supported operation aside from forward pass is 
           mutation.
    """

    def __init__(self, layers, active_funcs, mutation_rate, mutation_mag=1):

        self._mutation_rate = mutation_rate
        self._layers = layers
        self._active_funcs = active_funcs
        self._mutation_mag - mutation_mag
        
        self.num_inputs = layers[0]
        self.num_outputs = layers[-1]

        self.model = Sequential()
        self.model.add(Dense(layers[2], activation=active_funcs[0], input_dim=self.num_inputs))

        for i in range(1,len(layers)):
            self.model.add(Dense(layers[i], activation=active_funcs[i-1]))

    def mutate(self):
        """ Add, according to some mutation rate, gaussian noise with zero mean and
            unit variance scaled by a mutation magnitude, to weights in network.

        Mutates:
            self.model: For each layer, possibly adds noise to each neuron
        """
        for layer in self.model.layers:
            weights = np.array(layer.get_weights())
            n = weights.size

            N = np.random.randn(n) * self.mutation_mag
            B = np.random.rand(n) > (1 - self.mutation_rate)

            M = np.multiply(N,B)
            layer.set_weights(np.add(weights, M))

    def deep_copy(self):
        """ Copies FeedForwardNeuralNetwork, deep copying over the weight matrix
        
        Returns: 
            FeedForwardNeuralNetwork.
        """
        net = FeedForwardNeuralNet(self._layers, self._active_funcs,
                                   self._mutation_rate, self._mutation_mag)
        net.model.set_weights(np.copy(self.get_weights()))
        return net

    def copy_weights(self, other):
        """ Copy other network's weights to this network. 

        Args:
            other (either keras network or FFNN)

        Mutates:
            self.model: Sets weights for all layers.
        """
        self.model.set_weights(other.get_weights())
        
    def get_weights(self):
        """ Wrapper for model's get_weights function.  """
        return self.model.get_weights()

    def predict(self, inp, batch_size=1):
        """ Wrapper for model's predict function. """
        return self.model.predict(inp, batch_size=batch_size)
