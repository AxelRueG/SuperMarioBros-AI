import retro
from utils import SMB
from config import Config
from genetic_algorithm.genetico import Genetico

class Game:

    def __init__(self):
        self.i = 0
        self.done = False
        self.config = Config('settings.json')

        self.genetico = Genetico(self.config)

        # @TODO: hacer que esto sea opcional segun config
        # self.mario = load_mario(op.abspath("individuals/test/"), "best_ind_gen1")

    def run(self):
        env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
        obs = env.reset()
        while True:
            # @TODO: hacer que la grafica sea opcional segun config
            env.render()

            ram = env.get_ram()                                # estado actual del juego
            tiles = SMB.get_tiles(ram)                              # procesa la grilla

            self.genetico.player.update(ram, tiles)
            
            obs, rew, done, info = env.step(self.genetico.player.buttons_to_press)
            self.genetico.player.calculate_fitness()

            if not self.genetico.player.is_alive:
                self.genetico.next_individuo()
                env.reset()

if __name__ == "__main__":

    game = Game()
    game.run()