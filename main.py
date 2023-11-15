import retro, os.path as op
from utils import SMB
from config import Config
from genetic_algorithm.genetico import Genetico
from typing import Optional
from genetic_algorithm.utils import load_mario 

class Game:

    def __init__(self, individuo_file: Optional[str] = None):
        self.i = 0
        self.done = False
        self.config = Config('settings.json')

        self.replay = True if individuo_file else False

        if (self.replay):
            elems = individuo_file.split('/')
            folder = '/'.join(elems[:len(elems)-1])
            individuo_folder = elems[-1]
            self.mario = load_mario(op.abspath(folder), individuo_folder)
            self.config = self.mario.config # setteo la configuracion guardada de ese jugador
        else:
            self.genetico = Genetico(self.config)

    def run(self):
        env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
        obs = env.reset()
        while True:
            if self.config.Graphics['enable']: env.render()

            ram = env.get_ram()                                # estado actual del juego
            tiles = SMB.get_tiles(ram)                         # procesa la grilla

            if not self.replay:
                self.genetico.player.update(ram, tiles)
                self.genetico.player.calculate_fitness()

                obs, rew, done, info = env.step(self.genetico.player.buttons_to_press)
                if not self.genetico.player.is_alive:
                    self.genetico.next_individuo()
                    env.reset()
        
            else:

                self.mario.update(ram, tiles)
                self.mario.calculate_fitness()

                obs, rew, done, info = env.step(self.mario.buttons_to_press)
                if not self.mario.is_alive:
                    env.close()
        env.close()

if __name__ == "__main__":

    #game = Game('./individuals/test/best_ind_gen10') #para ver la run x
    game = Game() #para entrenar 


    game.run()