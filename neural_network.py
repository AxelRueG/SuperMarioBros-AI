import numpy as np
from typing import List, Callable, NewType, Optional

# ----- Type of activation function -----------------------------------------------------------------
ActivationFunction = NewType('ActivationFunction', Callable[[np.ndarray], np.ndarray])


# --------------------------------------------------------------------------------------------------
#       activation functions 
# --------------------------------------------------------------------------------------------------
def get_activation_by_name(name: str) -> ActivationFunction:
    if name == 'relu':
        return ActivationFunction(lambda X: np.maximum(0, X))
    elif name == 'linear':
        return ActivationFunction(lambda X: X)
    else: # sigmoide
        return ActivationFunction(lambda X: 1.0 / (1.0 + np.exp(-X)))


# --------------------------------------------------------------------------------------------------
#       NN - eval
# --------------------------------------------------------------------------------------------------
class FeedForwardNetwork(object):
    def __init__(self,
                 layer_nodes: List[int],
                 hidden_activation: ActivationFunction,
                 output_activation: ActivationFunction,
                 seed: Optional[int] = None):
        self.params = {}
        self.layer_nodes = layer_nodes # [in/out de cada capa]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.out = None

        self.rand = np.random.RandomState(seed)

        # crear weights and bias
        for l in range(1, len(self.layer_nodes)):
            # pesos [W<layer_num>]
            self.params['W' + str(l)] = np.random.uniform(-1, 1, size=(self.layer_nodes[l], self.layer_nodes[l-1] + 1))
            
            ## @NOTE: si vamos a graficar la NN lo necesitamos 
            ## salidas [y<layer_num>]  
            # self.params['y' + str(l)] = None
        
        
    def feed_forward(self, X: np.ndarray) -> np.ndarray:  
        y_prev = np.hstack((X.reshape(X.shape[0],),-1))
        L = len(self.layer_nodes) - 1  # len(self.params) // 2

        # Feed hidden layers
        for l in range(1, L):
            W = self.params['W' + str(l)]
            y_prev = self.hidden_activation(np.dot(W, y_prev))
            y_prev = np.hstack((y_prev,-1))

        # Feed output
        W = self.params['W' + str(L)]
        self.out = self.output_activation(np.dot(W, y_prev))

        return self.out

# NN = FeedForwardNetwork([2,2,1],
#                         get_activation_by_name('sigmoide'),
#                         get_activation_by_name('sigmoide'))

# print(NN.feed_forward(np.array([1,1])))