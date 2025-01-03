## Contributing 

To implement a new functionality in the `aten` library, you must 
1. Add the class or function header in `aten/src/Tensor.h` 
2. Add the implementation in the correct file (or create a new one) in `aten./*Tensor/*.cpp`. Make sure to update `aten/bindings/CMakeLists.txt` if needed.
3. Add its pybindings (if a public function that will be used in `ember`) in `aten/bindings/*bindings.cpp`. Make sure to update `aten/bindings/CMakeLists.txt` if needed. 
4. Add relevant C++ tests in `aten/test/`.  
5. Not necessary, but it's good to test it out on a personal script for a sanity check.  
6. Add to the stub files in `ember/aten/*.pyi`. 
7. Add Python tests in `test/`. 
8. If everything passes, you can submit a pull request. 
