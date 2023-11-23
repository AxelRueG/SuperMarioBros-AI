# Algoritmo neuroevolutivo para jugar Super Mario Bros de manera automática

Para hacer andar el proyecto necesita tener instalado python 3.8 (preferentemente) o 3.9

para instalar las dependencias necesarias ejecutar:

~~~ bash
pip install -r requeriments.txt
~~~

Para correr el proyecto tenemos el archivo main tenemos, donde se especifica lo siguiente: 

~~~ python
if __name__ == "__main__":

    ## ---- SI QUEREMOS REPETIR UN INDIVIDUO -----
    # carpeta donde esta guardado el individuo
    folder = './individuals/test1/best_ind_gen137'
    game = Game(folder) 
    game.run()


    ## ---- SI QUEREMOS ENTRENAR (por temas de paralelismo en config.Graphics.enable == false) ----
    # game = Game()
    # tic = time.time()
    # game.trn()
    # toc = time.time()
    # print(f"timepo de entreno {toc-tic}")
~~~

Tambien tenemos el archivo de configuracion _settings.json_, donde se podra cambiar variables como la taza de mutacion la porbabilidad de cruza, la arquitectura de la capa de entrada y oculta del MLP como asi la funcion de activacion de estas capas.