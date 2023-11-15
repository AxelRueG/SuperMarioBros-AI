import numpy as np
from neural_network import FeedForwardNetwork, get_activation_by_name
from utils import SMB, StaticTileType, EnemyType
from config import Config
from typing import Optional, Dict

class Player:
    def __init__(self,
                 config: Config,
                 chromosome: Optional[Dict[str, np.ndarray]] = None
                 ):

        # ---- variables de configuracion ----
        self.config = config

        # ---- configurar la NN ----
        self.start_row, self.viz_width, self.viz_height = self.config.NeuralNetwork["input_dims"]

        if self.config.NeuralNetwork["encode_row"]:
            # codifica un vector binario con un solo 1 en la posicion donde se encuentra mario
            num_inputs = self.viz_width * self.viz_height + self.viz_height
        else:
            num_inputs = self.viz_width * self.viz_height
        
        self.inputs_as_array = np.zeros((num_inputs, 1))
        self.nn_arch = [num_inputs]                                                  # nodos de entrada
        self.nn_arch.extend(self.config.NeuralNetwork["hidden_layer_architecture"])  # nodos de la capa oculata
        self.nn_arch.append(6)                              # las 6 salidas : abajo, arriba, izq, derecha, a, b

        self.nn = FeedForwardNetwork(self.nn_arch,
                                     get_activation_by_name(self.config.NeuralNetwork["hidden_node_activation"]),
                                     get_activation_by_name(self.config.NeuralNetwork["output_node_activation"])
                                    )

        # si estan seteados los cromosomas, los toma
        if chromosome:
            self.nn.params = chromosome
        
        # ---- variables de estado del jugador ----
        self.fitness = 0                # Overall fitness
        self.frames_since_progress = 0  # Número de frames desde que Player avanzó hacia la meta
        self.frames = 0                 # Número de frames que Player ha estado vivo
        self.is_alive = True
        self.x_dist = None
        self.game_score = None
        self.did_win = False

        # Esto es principalmente para "ver" a Player ganar. 
        self.enable_additional_time  = self.config.General["allow_additional_time_for_flagpole"]
        self.additional_timesteps = 0
        self.max_additional_timesteps = int(60*2.5)

        # ---- Acciones del jugador ----
        # set de teclas                    B, NULL, SELECT, START, U, D, L, R, A
        # index                            0  1     2       3      4  5  6  7  8
        self.buttons_to_press = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)

        # Solo permito U, D, L, R, A, B y esos son los índices en los que se generará la salida.
        # Necesitamos un mapeo desde la salida a las claves anteriores
        self.ouput_to_keys_map = {
            0: 4,  # U
            1: 5,  # D
            2: 6,  # L
            3: 7,  # R
            4: 8,  # A
            5: 0   # B
        }
        self.farthest_x = 0
    
    #calculo el fitness del frame
    def calculate_fitness(self):
        '''
        args:
        - frames:     Number of frames that Player has been alive for
        - distance:   Total horizontal distance gone through the level
        - did_win:    True/False if Player beat the level
        '''
        frames = self.frames
        distance = self.x_dist        
        did_win = self.did_win
        
        self.fitness = max((distance**1.8) - (frames**1.5) + (min(max(distance-50, 0), 1) * 2500) + (did_win * 1e6), 0.00001)

    #esto arma la grilla para detectar que elementos se encuentran en ella
    def set_input_as_array(self, ram, tiles) -> None:
        mario_row, mario_col = SMB.get_mario_row_col(ram)
        arr = []
        #(start_row, width, height)-> (4, 7, 10)  donde width y height van a conformar la malla de pixeles
        for row in range(self.start_row, self.start_row + self.viz_height):
            for col in range(mario_col, mario_col + self.viz_width):
                try:
                    t = tiles[(row, col)]
                    if isinstance(t, StaticTileType):
                        if t.value == 0:
                            arr.append(0)
                        else:
                            arr.append(1)
                    elif isinstance(t, EnemyType):
                        arr.append(-1)
                    else:
                        raise Exception("This should never happen")
                except:
                    t = StaticTileType(0x00)
                    arr.append(0) # Empty

        self.inputs_as_array[:self.viz_height*self.viz_width, :] = np.array(arr).reshape((-1,1))
        if self.config.NeuralNetwork["encode_row"]:
            # Asignar one-hot para mario row (marcamos la posicion en y de mario con un bit)
            row = mario_row - self.start_row
            one_hot = np.zeros((self.viz_height, 1))
            if row >= 0 and row < self.viz_height:
                one_hot[row, 0] = 1
            self.inputs_as_array[self.viz_height*self.viz_width:, :] = one_hot.reshape((-1, 1))

    def update(self, ram, tiles) -> bool:
        """
        Es el principal update para mario.
        toma los imputs del area del entorno y se alimenta mediante la red neuronal
        lo que devuelve es si mario esta vivo (True) o no (False)
        """
        #si mario esta vivo, aumenta el frame, se setea la distancia que se recorrio en el eje x, y se guarda el score 
        #TODO quitar lo del score
        if self.is_alive:
            self.frames += 1
            self.x_dist = SMB.get_mario_location_in_level(ram).x
            self.game_score = SMB.get_mario_score(ram)
            
            # si llegamos a la meta, printeamos un mensaje por consola, es necesario correr el demo con --debug
            if ram[0x001D] == 3:
                self.did_win = True
                if self.config.General['debug']: print(f'GANAMOS!!!')
                if not self.enable_additional_time:
                    self.is_alive = False
                    return False
            
            # actualizo la mejor distancia si es que se llego mas lejos y reseteo la actual            
            if self.x_dist > self.farthest_x:
                self.farthest_x = self.x_dist
                self.frames_since_progress = 0
            else:
                self.frames_since_progress += 1

            #por si me paso del tiempo limite 
            if self.enable_additional_time and self.did_win:
                self.additional_timesteps += 1
            
            #si me paso del maximo de timesteps, mato a mario 
            if self.enable_additional_time and self.additional_timesteps > self.max_additional_timesteps:
                self.is_alive = False
                return False
            
            elif not self.did_win and self.frames_since_progress > 60*3:
                self.is_alive = False
                return False            
        else:
            return False

        # por si caemos al vacio
        if ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2:
            self.is_alive = False
            return False

        self.set_input_as_array(ram, tiles)

        
        # ------------------------------------------------------------------------------------------
        #                           calculo la siguiente accion
        # ------------------------------------------------------------------------------------------
        output = self.nn.feed_forward(self.inputs_as_array)
        threshold = np.where(output > 0.5)[0]
        self.buttons_to_press.fill(0)  # limpio botones

        # seteo los botones
        for b in threshold:
            self.buttons_to_press[ self.ouput_to_keys_map[b]] = 1

        return True

