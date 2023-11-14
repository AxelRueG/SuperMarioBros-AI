import retro
import os.path as op
import numpy as np

from mario import Player, load_mario
from utils import SMB

class Game:

    def __init__(self):
        self.i = 0
        self.done = False

        self.mario = load_mario(op.abspath("individuals/test/"), "best_ind_gen1")


    def run(self):
        env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
        obs = env.reset()
        while True:
            env.render()

            ram = env.get_ram()                                # estado actual del juego
            tiles = SMB.get_tiles(ram)                              # procesa la grilla

            self.mario.update(ram, tiles)

            obs, rew, done, info = env.step(self.mario.buttons_to_press)
            
            if done:
                print(info)
                env.close()
                # env.reset()

if __name__ == "__main__":
    game = Game()
    game.run()