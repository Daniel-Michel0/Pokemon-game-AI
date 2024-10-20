# Setup
Se recomienda crear un ambiente virtual antes de instalar el proyecto.
1. Instalar [Node.js v10+](https://nodejs.org/en/).
2. Clonar el repositorio de pokémon showdown y configurarlo.
    > git clone https://github.com/smogon/pokemon-showdown.git
    cd pokemon-showdown
    npm install
    cp config/config-example.js config/config.js
    node pokemon-showdown start --no-security
3. Instalar las dependencias.
    > pip install -r requirements.txt


# Uso
1. Correr el servidor local de pokémon showdown.
    > cd pokemon-showdown
      node pokemon-showdown start --no-security
2. Ejecutar el programa.
    > python poke-rl.py