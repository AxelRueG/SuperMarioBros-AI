import retro, os.path as op
from utils import SMB
from config import Config
from genetic_algorithm.genetico import Genetico
from typing import Optional
from genetic_algorithm.utils import load_mario 

from joblib import Parallel, delayed
import time

class Game:

    def __init__(self, individuo_file: Optional[str] = None):
        self.i = 0
        self.done = False
        self.config = Config('settings.json')

        self.replay = True if individuo_file else False

        self.mario = None
        self.genetico = None

        # Cargo los jugadores
        if (self.replay):
            elems = individuo_file.split('/')
            folder = '/'.join(elems[:len(elems)-1])
            individuo_folder = elems[-1]
            self.mario = load_mario(op.abspath(folder), individuo_folder)
            self.config = self.mario.config # setteo la configuracion guardada de ese jugador
        else:
            self.genetico = Genetico(self.config)

    def run(self, i: int = 0):
        env = retro.make(game='SuperMarioBros-Nes', state=f'Level{self.config.General["level"]}')
        obs = env.reset()

        # Compruevo si es una repeticion
        if self.replay:
            mario = self.mario
        else:
            mario = self.genetico.poblacion.individuals[i]

        while True:
            # Inicio el juego
            if self.config.Graphics['enable']: env.render()

            ram = env.get_ram()                                # estado actual del juego
            tiles = SMB.get_tiles(ram)                         # procesa la grilla

            mario.update(ram, tiles)
            mario.calculate_fitness()

            obs, rew, done, info = env.step(mario.buttons_to_press)
            if not mario.is_alive:
                try:
                    env.close()
                except ():
                    pass

                if self.config.General['debug']: 
                    print(f'individuo actual {i} fitness {mario.fitness} distance {mario.farthest_x}')

                return mario
        env.close()


    def trn(self):

        # esto es un aproximado del fitness de un individuo ganador
        # o podriamos hacerlo por epocas
        while self.genetico.best_fitness < 3037415:
            
            ## ---- Evaluar jugadores en paralelo [Multinucleo] ----
            # tic = time.time()
            res = Parallel(n_jobs=-1) \
                (delayed(self.run)(individuo) for individuo in range(self.genetico.poblacion.num_individuals))
            # toc = time.time()
            # print(f"timepo de entreno {toc-tic}")

            ## ---- Evaluar en [Mononucleo] ----
            # for individuo in range(self.genetico.poblacion.num_individuals):
            #     self.run(individuo)

            # ---- Actualizar Poblaccion -----------------------------------------------------------
            self.genetico.poblacion.individuals = res
            self.genetico.next_generation()
            

if __name__ == "__main__":

    ## SI QUEREMOS REPETIR UN INDIVIDUO
    # game = Game('./individuals/test1/best_ind_gen5')
    # game.run()

    ## SI QUEREMOS ENTRENAR (por temas de paralelismo en config.Graphics.enable == false)
    game = Game()
    game.trn()