import os.path as op
import json

# --------------------------------------------------------------------------------------------------
#           Clase que almacena toda variables de configuracion
# --------------------------------------------------------------------------------------------------
class Config(object):
    def __init__(self, filename: str):
        self.filename = filename
        
        if not op.isfile(self.filename):
            raise Exception('No file found named "{}"'.format(self.filename))
        
        # Leer el archivo JSON y usarlo como parametros del objeto
        with open(self.filename, 'r') as archivo:
            self.__dict__.update(json.load(archivo))

