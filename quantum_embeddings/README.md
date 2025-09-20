# Tesis

Este repositorio utiliza [uv](https://astral.sh/blog/uv) para la gestión de dependencias y el empaquetado.

## Instalación de dependencias

Una vez instalado uv, puedes instalar las dependencias del proyecto con los siguientes comandos:


- Con uv init inicializas el proyecto, crea un pyproject.toml vacio. 
- Antes de correr uv sync, correr el comando `uv pip install "qiskit-machine-learning @ packages\qiskit_machine_learning-0.9.0-py3-none-any.whl"`
- Con uv sync, se genera el .venv y descarga lo listado en el pyproject.toml. 
- Con uv add <nombre_de_dependencia> agrega la dependencia al pyproject.toml.
- Con uv pip install <nombre_del_paquete> instalas cosas en el .venv. 
- Con uv run main.py corres el main. Sino, podes hacer source .venv/bin/activate y correr el main con python main.py
