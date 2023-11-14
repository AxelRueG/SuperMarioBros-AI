import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
import os
import csv
import json

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from neural_network import FeedForwardNetwork, ActivationFunction, get_activation_by_name
from utils import SMB, StaticTileType, EnemyType
from config import Config

class Mario(Individual):
    def __init__(self,
                 config: Config,
                 chromosome: Optional[Dict[str, np.ndarray]] = None,
                 hidden_layer_architecture: List[int] = [12, 9],
                 hidden_activation: Optional[ActivationFunction] = 'relu',
                 output_activation: Optional[ActivationFunction] = 'sigmoid',
                 lifespan: Union[int, float] = np.inf,
                 name: Optional[str] = None,
                 debug: Optional[bool] = False,
                 ):
        

        self.config = config

        self.lifespan = lifespan
        self.name = name
        self.debug = debug

        self._fitness = 0  # Overall fitness
        self._frames_since_progress = 0  # Número de frames desde que Mario avanzó hacia la meta
        self._frames = 0  # Número de frames que Mario ha estado vivo
        
        #seteo la config de la arquitectura de la capa oculta y la manera en la que se activan las neuronas
        self.hidden_layer_architecture = self.config.NeuralNetwork.hidden_layer_architecture
        self.hidden_activation = self.config.NeuralNetwork.hidden_node_activation
        self.output_activation = self.config.NeuralNetwork.output_node_activation

        self.start_row, self.viz_width, self.viz_height = self.config.NeuralNetwork.input_dims

        
        if self.config.NeuralNetwork.encode_row:
            num_inputs = self.viz_width * self.viz_height + self.viz_height
        else:
            num_inputs = self.viz_width * self.viz_height
        # print(f'num inputs:{num_inputs}')
        
        self.inputs_as_array = np.zeros((num_inputs, 1))
        self.network_architecture = [num_inputs]                          # nodos de entrada
        self.network_architecture.extend(self.hidden_layer_architecture)  # nodos de la capa oculata
        self.network_architecture.append(6)                        # las 6 salidas : abajo, arriba, izq, derecha, a, b

        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation)
                                         )

        # si estan seteados los cromosomas, los toma
        if chromosome:
            self.network.params = chromosome
        
        self.is_alive = True
        self.x_dist = None
        self.game_score = None
        self.did_win = False

        # Esto es principalmente para "ver" a Mario ganar. 
        self.allow_additional_time  = self.config.Misc.allow_additional_time_for_flagpole
        self.additional_timesteps = 0
        self.max_additional_timesteps = int(60*2.5)
        self._printed = False

        # set de teclas                    B, NULL, SELECT, START, U, D, L, R, A
        # index                            0  1     2       3      4  5  6  7  8
        self.buttons_to_press = np.array( [0, 0,    0,      0,     0, 0, 0, 0, 0], np.int8)
        self.farthest_x = 0

    #getter del fitness
    @property
    def fitness(self):
        return self._fitness
    
    #calculo el fitness del frame
    def calculate_fitness(self):
        frames = self._frames
        distance = self.x_dist
        score = self.game_score

        self._fitness = fitness_func(frames, distance, score, self.did_win)

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
        if self.config.NeuralNetwork.encode_row:
            # Assign one-hot for mario row
            row = mario_row - self.start_row
            one_hot = np.zeros((self.viz_height, 1))
            if row >= 0 and row < self.viz_height:
                one_hot[row, 0] = 1
            self.inputs_as_array[self.viz_height*self.viz_width:, :] = one_hot.reshape((-1, 1))

    def update(self, ram, tiles, buttons, ouput_to_buttons_map) -> bool:
        """
        Es el principal update para mario.
        toma los imputs del area del entorno y se alimenta mediante la red neuronal
        lo que devuelve es si mario esta vivo (True) o no (False)
        """
        #si mario esta vivo, aumenta el frame, se setea la distancia que se recorrio en el eje x, y se guarda el score 
        #TODO quitar lo del score
        if self.is_alive:
            self._frames += 1
            self.x_dist = SMB.get_mario_location_in_level(ram).x
            self.game_score = SMB.get_mario_score(ram)
            
            # si llegamos a la meta, printeamos un mensaje por consola, es necesario correr el demo con --debug
            if ram[0x001D] == 3:
                self.did_win = True
                if not self._printed and self.debug:
                    name = 'Mario '
                    name += f'{self.name}' if self.name else ''
                    print(f'{name} won')
                    self._printed = True
                if not self.allow_additional_time:
                    self.is_alive = False
                    return False
            
            # actualizo la mejor distancia si es que se llego mas lejos y reseteo la actual            
            if self.x_dist > self.farthest_x:
                self.farthest_x = self.x_dist
                self._frames_since_progress = 0
            else:
                self._frames_since_progress += 1

            #por si me paso del tiempo limite 
            if self.allow_additional_time and self.did_win:
                self.additional_timesteps += 1
            
            #si me paso del maximo de timesteps, mato a mario 
            if self.allow_additional_time and self.additional_timesteps > self.max_additional_timesteps:
                self.is_alive = False
                return False
            
            elif not self.did_win and self._frames_since_progress > 60*3:
                self.is_alive = False
                return False            
        else:
            return False

        # por si caemos al vacio
        if ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2:
            self.is_alive = False
            return False

        self.set_input_as_array(ram, tiles)

        
        # --------------------------------------------------------------------------------------------------
        #                           calculo la salida
        # --------------------------------------------------------------------------------------------------
        output = self.network.feed_forward(self.inputs_as_array)
        threshold = np.where(output > 0.5)[0]
        self.buttons_to_press.fill(0)  # limpio botones

        # seteo los botones
        for b in threshold:
            self.buttons_to_press[ouput_to_buttons_map[b]] = 1

        return True
    
#esta funcion guarda en un bin los pesos y el bias del modelo
def save_mario(population_folder: str, individual_name: str, mario: Mario) -> None:
    # Make population folder if it doesnt exist
    if not os.path.exists(population_folder):
        os.makedirs(population_folder)

    # ---- guardo settings.config ----
    if 'settings.json' not in os.listdir(population_folder):
        with open(os.path.join(population_folder, 'settings.json'), 'w') as config_file:
            json.dump(mario.config._config, config_file, indent=2)
    
    # ---- crea el directorio para el individual ----
    individual_dir = os.path.join(population_folder, individual_name)
    os.makedirs(individual_dir)

    # ---- Guarda los pesos ----
    L = len(mario.network.layer_nodes)
    for l in range(1, L):
        w_name = 'W' + str(l)
        weights = mario.network.params[w_name]
        np.save(os.path.join(individual_dir, w_name), weights)
    
#carga el archivo del individual
def load_mario(population_folder: str, individual_name: str, config: Optional[Config] = None) -> Mario:
    # se asegura que exista dentro de la carpeta population
    if not os.path.exists(os.path.join(population_folder, individual_name)):
        raise Exception(f'{individual_name} not found inside {population_folder}')

    # cargo la config
    if not config:
        settings_path = os.path.join(population_folder, 'settings.json')
        config = None
        try:
            config = Config(settings_path)
        except:
            raise Exception(f'settings.config not found under {population_folder}')

    chromosome: Dict[str, np.ndarray] = {}

    # tomo los archivos .npy , como por ej: W1.npy, b1.npy, etc. y los cargo en el cromosoma
    for fname in os.listdir(os.path.join(population_folder, individual_name)):
        extension = fname.rsplit('.npy', 1)
        if len(extension) == 2:
            param = extension[0]
            chromosome[param] = np.load(os.path.join(population_folder, individual_name, fname))
        
    mario = Mario(config, chromosome=chromosome)
    return mario

#calculo las estadisticas 
def _calc_stats(data: List[Union[int, float]]) -> Tuple[float, float, float, float, float]:
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    _min = float(min(data))
    _max = float(max(data))

    return (mean, median, std, _min, _max)

#guardo las estadisticas
def save_stats(population: Population, fname: str):
    directory = os.path.dirname(fname)
    #por si no existe el directorio lo creo
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = fname
    #cargo la informacion de los individuos de la poblacion
    frames = [individual._frames for individual in population.individuals]
    max_distance = [individual.farthest_x for individual in population.individuals]
    fitness = [individual.fitness for individual in population.individuals]
    wins = [sum([individual.did_win for individual in population.individuals])]

    write_header = True
    #si ya tengo hecho el header, no lo sobreescribo mas adelante
    if os.path.exists(f):
        write_header = False

    trackers = [('frames', frames),
                ('distance', max_distance),
                ('fitness', fitness),
                ('wins', wins)
                ]

    stats = ['mean', 'median', 'std', 'min', 'max']

    header = [t[0] + '_' + s for t in trackers for s in stats]

    with open(f, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header, delimiter=',')
        if write_header:
            writer.writeheader()

        row = {}
        # creo una fila para agregarla al csv
        for tracker_name, tracker_object in trackers:
            curr_stats = _calc_stats(tracker_object)
            for curr_stat, stat_name in zip(curr_stats, stats):
                entry_name = '{}_{}'.format(tracker_name, stat_name)
                row[entry_name] = curr_stat

        # escribo la fila
        writer.writerow(row)

#cargo las estadisticas
def load_stats(path_to_stats: str, normalize: Optional[bool] = False):
    data = {}

    fieldnames = None
    trackers_stats = None
    trackers = None
    stats_names = None

    #abro el archivo en modo lectura
    with open(path_to_stats, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        fieldnames = reader.fieldnames
        trackers_stats = [f.split('_') for f in fieldnames]
        trackers = set(ts[0] for ts in trackers_stats)
        stats_names = set(ts[1] for ts in trackers_stats)
        
        for tracker, stat_name in trackers_stats:
            if tracker not in data:
                data[tracker] = {}
            
            if stat_name not in data[tracker]:
                data[tracker][stat_name] = []

        for line in reader:
            for tracker in trackers:
                for stat_name in stats_names:
                    value = float(line['{}_{}'.format(tracker, stat_name)])
                    data[tracker][stat_name].append(value)
    
    # normalizo los stats si es que se requiere
    if normalize:
        factors = {}
        for tracker in trackers:
            factors[tracker] = {}
            for stat_name in stats_names:
                factors[tracker][stat_name] = 1.0

        for tracker in trackers:
            for stat_name in stats_names:
                max_val = max([abs(d) for d in data[tracker][stat_name]])
                if max_val == 0:
                    max_val = 1
                factors[tracker][stat_name] = float(max_val)

        for tracker in trackers:
            for stat_name in stats_names:
                factor = factors[tracker][stat_name]
                d = data[tracker][stat_name]
                data[tracker][stat_name] = [val / factor for val in d]

    return data

#determino y devuelvo la cantidad de entradas de la red neuronal
def get_num_inputs(config: Config) -> int:
    _, viz_width, viz_height = config.NeuralNetwork.input_dims
    if config.NeuralNetwork.encode_row:
        num_inputs = viz_width * viz_height + viz_height
    else:
        num_inputs = viz_width * viz_height
    return num_inputs

#determino el numero de parametros entrenables 
def get_num_trainable_parameters(config: Config) -> int:
    num_inputs = get_num_inputs(config)
    hidden_layers = config.NeuralNetwork.hidden_layer_architecture
    num_outputs = 6  # abajo, arriba, izq, derecha, a, b

    layers = [num_inputs] + hidden_layers + [num_outputs]
    num_params = 0
    for i in range(0, len(layers)-1):
        L      = layers[i]
        L_next = layers[i+1]
        num_params += L*L_next + L_next

    return num_params

def fitness_func (frames, distance, game_score, did_win): 
    '''
    args:
    - frames:     Number of frames that Mario has been alive for
    - distance:   Total horizontal distance gone through the level
    - game_score: Actual score Mario has received in the level through power-ups, coins, etc.
    - did_win:    True/False if Mario beat the level

    return:
    - fitnes
    '''
    return max((distance**1.8) - (frames**1.5) + (min(max(distance-50, 0), 1) * 2500) + (did_win * 1e6), 0.00001)