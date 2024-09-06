# ember

Lightweight ML library for my personal use with C++ and Python. 

Copy the `pybind11` repo into `_CPP` directory. 
```
cd ember/_CPP 
git submodule add https://github.com/pybind/pybind11.git
git submodule update --init
```
Build the `_CPP` library for low level tensor operations with the following. Make sure `cmake` is installed. 
``` 
cd ember/_CPP 
mkdir build && cd build 
cmake ..
make 
```
Copy the `.so` file to `ember`. Then it is accessible by the python modules. Finally, run `pip install -e .` in the root directory to install the package. 
