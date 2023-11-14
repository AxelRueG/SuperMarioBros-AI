import os
import json
from typing import Any, Dict

# --------------------------------------------------------------------------------------------------
#           Genera un objeto a partir de un json
# --------------------------------------------------------------------------------------------------
class DotNotation(object):
    def __init__(self, d: Dict[Any, Any]):
        for k in d:
            # If the key is another dictionary, keep going
            if isinstance(d[k], dict):
                self.__dict__[k] = DotNotation(d[k])
            # If it's a list or tuple then check to see if any element is a dictionary
            elif isinstance(d[k], (list, tuple)):
                l = []
                for v in d[k]:
                    if isinstance(v, dict):
                        l.append(DotNotation(v))
                    else:
                        l.append(v)
                self.__dict__[k] = l
            else:
                self.__dict__[k] = d[k]
    
    def __getitem__(self, name) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self)

# --------------------------------------------------------------------------------------------------
#           Clase que almacena toda variables de configuracion
# --------------------------------------------------------------------------------------------------
class Config(object):
    def __init__(self, filename: str):
        self.filename = filename
        
        if not os.path.isfile(self.filename):
            raise Exception('No file found named "{}"'.format(self.filename))

        # with open(self.filename) as f:
        #     self._config_text_file = f.read()
        # Leer el archivo JSON
        with open(self.filename, 'r') as archivo:
            self._config = json.load(archivo)

        dot_notation = DotNotation(self._config)
        self.__dict__.update(dot_notation.__dict__)
