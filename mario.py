import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any, List
import os
import csv
import json

from genetic_algorithm.individual import Player
from genetic_algorithm.population import Population
from config import Config
    
#esta funcion guarda en un bin los pesos y el bias del modelo
def save_mario(population_folder: str, individual_name: str, mario: Player) -> None:
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
def load_mario(population_folder: str, individual_name: str, config: Optional[Config] = None) -> Player:
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
        
    mario = Player(config, chromosome=chromosome)
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