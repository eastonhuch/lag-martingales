# lag-martingales


Build the Docker container with the following command: `docker build -t lag-martingales .`

Run the container interactively with this command:
```
docker run -it --rm \
  -p 8888:8888 \
  -v /path/to/lag-martingales:/home/jovyan/lag-martingales \
  lag-martingales
```

Compile the Cython code using this command: `python setup.py build_ext --inplace`.
Then execute the notebook `simulation.ipynb`.