# Progress 

  To do: 
  1. Add a template argument for Tensor dtype. 
  2. Store all tensors in heap to preserve them after stack is destroyed.  

  ✅ - Done
  ❌ - Not implemented
  🪧 - Don't need, either should not be accessed or is not necessary (e.g. due to inheritance)
  🚧 - In progress

  ## Aten BaseTensor 

  | C++ Method | PyBind Method | Status | C++ Tests | Python Tests | Stubs |
  |----------|----------|----------|----------|----------|----------|
  | `std::string type() const` | `type()` | ✅ | ✅ | 🪧 | 🪧    |    
  | `std::string dtype() const` | `dtype()` | ✅ | ✅ | 🪧     | 🪧    |  
  | `bool operator==(BaseTensor&)` | `__eq__()` | ✅ | ✅ | 🪧    | 🪧    |    
  | `bool operator!=(BaseTensor&)` | `__ne__()` | ✅ | ✅ | 🪧    | 🪧    |    
  | `double at(const std::vector<size_t>&) const` | `__getitem__()` | ✅ | ✅ | 🪧   | 🪧    | 
  | `double at(const std::vector<size_t>&)` | `__setitem__()` | ✅ | ✅ | 🪧   | 🪧   |    
  | `std::unique_ptr<BaseTensor> slice(const std::vector<Slice>&) const` | `__getitem__()` | ✅ | ✅ | 🪧   | 🪧   |  
  | `operator std::string const` | `__str__()` |  ✅ | ✅ | 🪧   | 🪧   |  
  | `operator std::string const` | `__repr__()` |  ✅ | ✅ | 🪧   | 🪧   |  

  ## Aten GradTensor 

  | C++ Method | PyBind Method | Status | C++ Tests | Python Tests | Stubs |
  |----------|----------|----------|----------|----------|----------|
  | `std::string type() const` | `type()` | ✅ | ✅  |  ✅   | ❌  |    
  | `std:: string dtype() const` | `dtype()` | ✅ | ✅ |  ✅   | ❌  |    
  | `bool operator==(GradTensor&)` | `__eq__()` | ✅ | ✅ | ✅   | ❌  |       
  | `bool operator!=(GradTensor&)` | `__ne__()` | ✅ | ✅ | ✅   | ❌  |
  | `double at(const std::vector<size_t>&) const` | `__getitem__()` | ✅ | ✅ | ✅ | ❌  |   
  | `double at(const std::vector<size_t>&)` | `__setitem__()` | ✅ | ✅ | ✅ | ❌ |
  | `std::unique_ptr<GradTensor> slice(const std::vector<Slice>&) const` | `__getitem__()` | ✅ | ✅ | ✅ | ❌  |
  | `size_t pivot() const` | `pivot()` | ✅ | ❌ | ❌ | ❌ | 
  | `GradTensor()` | `GradTensor()` | ✅ | ✅  | ✅  | ❌ | 
  | `GradTensor(std::vector<double>, std::vector<size_t>, size_t)` | `GradTensor(List[double], List[int], int)` | ✅ | ✅  | ✅  | ❌ | 
  | `GradTensor(std::vector<size_t>, size_t)` | `GradTensor(List[int], int)`| ✅ | ✅  | ✅  | ❌ | 
  | `GradTensor::eye(size_t, size_t)` | | ✅ | ✅  | ✅  | ❌ | 
  | `transpose()` |  `transpose()` | ✅ | ✅ | ❌ | ❌ | 
  | `GradTensor copy() const` | `copy()` | ✅ | ✅  | ✅  | ❌ |     
  | `GradTensor add(GradTensor&)` | `__add__(GradTensor)` | ✅ | ✅  | ✅  | ❌ | 
  | `Tensor add(Tensor&)` | `__add__(Tensor)` | ✅ | ✅  | ❌ | ❌ | 
  | `GradTensor add(ScalarTensor&)` | `__add__(ScalarTensor)` | ✅ | ❌ | ❌ | ❌ |      
  | `GradTensor add(double&)` | `__add__(float)` | ✅ | ❌ | ❌ | ❌ |      
  | `GradTensor sub(GradTensor&)` | `__sub__(GradTensor)` | ✅ | ✅  | ✅  | ❌ |      
  | `Tensor sub(Tensor&)` | `__sub__(Tensor)` | ✅ | ✅  | ❌ | ❌ | 
  | `GradTensor sub(ScalarTensor&)` | `__sub__(ScalarTensor)` | ✅ | ❌ | ❌ | ❌ |      
  | `GradTensor sub(double&)` | `__sub__(float)` | ✅ | ❌ | ❌ | ❌ |
  | `GradTensor mul(GradTensor&)` | `__mul__(GradTensor)` | ✅ | ✅  | ✅  | ❌ | 
  | `Tensor mul(Tensor&)` | `__mul__(Tensor)` | ✅ | ✅  | ❌ | ❌ | 
  | `GradTensor mul(ScalarTensor&)` | `__mul__(ScalarTensor)` | ✅ | ❌ | ❌ | ❌ |      
  | `GradTensor mul(double&)` | `__mul__(float)` | ✅ | ❌ | ❌ | ❌ | 
  | `GradTensor matmul(GradTensor&)` | `__matmul__(GradTensor)` | ✅ | ✅  | ✅  | ❌ | 

  ## Aten Tensor 

  | C++ Method | PyBind Method | Status | C++ Tests | Python Tests | Stubs |
  |----------|----------|----------|----------|----------|----------|
  | `std::string type() const` | `type()` | ✅ | ❌ | ❌ | ❌ | 
  | `std:: string dtype() const` | `dtype()` | ✅ | ❌ | ❌ | ❌ | 
  | `bool operator==(Tensor&)` | `__eq__()` | ✅ | ❌ | ❌ | ❌ |    
  | `bool operator!=(Tensor&)` | `__ne__()` | ✅ | ❌ | ❌ | ❌ |    
  | `double at(const std::vector<size_t>&) const` | `__getitem__()` | ✅ | ❌ | ❌ | ❌ |
  | `double at(const std::vector<size_t>&)` | `__setitem__()` |  ✅ | ❌ | ❌ | ❌ |
  | `std::unique_ptr<Tensor> slice(const std::vector<Slice>&) const` | `__getitem__()` | ✅ | ❌ | ❌ | ❌ |
  | `Tensor(std::vector<double>, std::vector<size_t>)` | `Tensor(List[float], List[int])` | ✅ | ✅  | ❌ | ❌ |     
  | `Tensor(std::vector<double>)` | `Tensor(List[float])` | ✅ | ✅  | ❌ | ❌ |    
  | `Tensor(std::vector<std::vector<double>>)` | `Tensor(List[List[float]])` | ✅ | ✅  | ❌ | ❌ |     
  | `Tensor(std::vector<std::vector<std::vector<double>>>);` | `Tensor(List[List[List[float]]])` | ✅ | ✅  | ❌ | ❌ |      
  | `static Tensor arange(int, int, int)` | `Tensor.arange(int, int, int)` | ✅ | ✅  | ❌ | ❌ |     
  | `static Tensor linspace(double, double, int)` | `Tensor.linspace(float, float, int)` | ✅ | ✅  | ❌ | ❌ |     
  | `static Tensor gaussian(std::vector<size_t> , double, double)` | `Tensor.gaussian(List[int], float, float)` | ✅ |  ✅  | ❌ | ❌ |     
  | `static Tensor uniform(std::vector<size_t> , double, double)` | `Tensor.uniform(List[int], int, int)` | ✅ | ✅ |   ❌ | ❌ |      
  | `static Tensor ones(std::vector<size_t>)` | `Tensor.ones(List[int])` | ✅ | ✅  | ❌ | ❌ |     
  | `static Tensor zeros(std::vector<size_t>)` | `Tensor.zeros(List[int])` | ✅ | ✅  | ❌ | ❌ |      
  | `void build_topo(Tensor* v, std::set<Tensor*>&, std::vector<Tensor*>&)` | 🪧 | ✅ | ❌| 🪧 |  🪧 |
  | `prev_` | `prev` | ✅   
  | `std::vector<Tensor*> backprop(bool)` | `backprop(bool)` | ✅| ❌ | ❌ | ❌ |     
  | `Tensor copy() const` | `copy()` | ✅ | ❌ | ❌ | ❌ |    
  | `Tensor add(Tensor&)` | `__add__(Tensor)` | ✅ | ✅  | ✅  | ❌ |
  | `Tensor add(GradTensor&)` | `__add__(GradTensor)` | ✅ | ❌ | ✅  | ❌ |
  | `Tensor add(ScalarTensor&)` | `__add__(ScalarTensor)` | ✅  | ❌ | ❌ | ❌ |
  | `Tensor add(double&)` | `__add__(float)` | ✅  | ❌ | ❌ | ❌ |
  | `Tensor sub(Tensor&)` | `__sub__(Tensor)` | ✅| ✅ | ✅  | ❌ |
  | `Tensor sub(GradTensor&)`  | `__sub__(GradTensor)` | ✅ | ❌ | ✅  | ❌ |
  | `Tensor sub(ScalarTensor&)` | `__sub__(ScalarTensor)` | ✅ | ❌ | ❌ | ❌ |
  | `Tensor sub(double&)` | `__sub__(float)` | ✅ | ❌ | ❌ | ❌ |
  | `Tensor mul(Tensor&)` | `__mul__(Tensor)` | ✅ | ❌ | ✅  | ❌ |
  | `Tensor mul(GradTensor&)` | `__mul__(GradTensor)` | ✅ | ❌ | ✅  | ❌ |
  | `Tensor mul(ScalarTensor&)` | `__mul__(ScalarTensor)` | ✅ | ❌ | ❌ | ❌ |
  | `Tensor mul(double&)` | `__mul__(float)` | ✅ | ❌ | ❌ | ❌ |
  | `Tensor exp(ScalarTensor&, double&)` | | ❌ | ❌ | ❌ | ❌ |
  | `Tensor exp(double&, double&)` | | ❌ | ❌ | ❌ | ❌ |
  | `Tensor log(ScalarTensor&, double&)` | | ❌ | ❌ | ❌ | ❌ |
  | `Tensor log(double&, double&)` | | ❌ | ❌ | ❌ | ❌ |
  | `Tensor matmul(Tensor&)` | `matmul(Tensor)` | ✅ | ❌ | ❌ | ❌ |
  | `Tensor matmul(Tensor&)` | `__matmul__(Tensor)` | ✅ | ❌ | ✅  | ❌ |
  | `Tensor sum(Tensor&)` | | 🚧 | ❌ | ❌ | ❌ |
  | `Tensor tranpose(const std::vector<size_t>&) const` | `transpose(List[int])` | ✅ | ❌ | ✅  | ❌ |
  | `Tensor concat(Tensor&, size_t)` | | ❌ | ❌ | ❌ | ❌ |
  | `Tensor sin()` |  | ❌ | ❌ | ❌ | ❌ |
  | `Tensor cos()` |  | ❌ | ❌ | ❌ | ❌ |
  | `Tensor tan()` |  | ❌ | ❌ | ❌ | ❌ |
  | `Tensor arcsin()` |  | ❌ | ❌ | ❌ | ❌ |
  | `Tensor arccos()` |  | ❌ | ❌ | ❌ | ❌ |
  | `Tensor arctan()` |  | ❌ | ❌ | ❌ | ❌ |
  | `Tensor relu()` |  | ❌ | ❌ | ❌ | ❌ |
  | `Tensor sigmoid()` |  | ❌ | ❌ | ❌ | ❌ |
  | `Tensor leaky_relu()` |  | ❌ | ❌ | ❌ | ❌ | 
  | `Tensor sum()` |  | ❌ | ❌ | ❌ | ❌ |
  | `Tensor mean()` |  | ❌ | ❌ | ❌ | ❌ |
  | `Tensor norm()` |  | ❌ | ❌ | ❌ | ❌ |


  ## ScalarTensor

  | C++ Method | PyBind Method | Status | C++ Tests | Python Tests | Docs |
  |----------|----------|----------|----------|----------|----------|
  | `std::string type() const` | `type()` |  | ✅  | 
  | `std:: string dtype() const` | `dtype()` |  | ✅ | 
  | `bool operator==(ScalarTensor&)` | `__eq__()` |  | ✅ | 
  | `bool operator!=(ScalarTensor&)` | `__ne__()` |  | ✅ | 
  | `double at(const std::vector<size_t>&) const` | `__getitem__()` |  | ✅ 
  | `double at(const std::vector<size_t>&)` | `__setitem__()` |  | ✅ 
  | `std::unique_ptr<ScalarTensor> slice(const std::vector<Slice>&) const` | `__getitem__()` |  | ✅   
  | `ScalarTensor copy() const` |  | ✅
  | `double item() const` |  | ✅
  | `Tensor add(Tensor&)` |  | ✅
  | `GradTensor add(GradTensor&)` |  | ✅
  | `Tensor sub(Tensor&)` |  | ✅
  | `GradTensor sub(GradTensor&)` |  | ✅
  | `Tensor mul(Tensor&)` |  | ✅
  | `GradTensor mul(GradTensor&)` |  | ✅


  ## Models 

