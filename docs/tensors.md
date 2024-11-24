Tensors are $N$-dimensional arrays that are used to represent tabular data or model parameters. They are both derived from the `BaseTensor` abstract class, which support the very minimal functionalities that all tensors should have. 

The hierarchy is 

    ```
    BaseTensor 
        Tensor 
            ScalarTensor (TBI)
            DenseTensor  (TBI)
            SparseTensor (TBI)
        GradTensor
    ``` 

All the attributes and methods that are supported by all classes can be found in `aten/src/Tensor.h`. 

# BaseTensor 

### Attributes

`std::vector<double> storage_`
- A contiguous vector  of doubles that store the state of the tensor. 

`std::vector<size_t> shape_`
- The shape of the tensor, where the product of the shape elements should match the length of `storage_`. 

### Methods


`virtual std::string type() const { return "BaseTensor"; }`
- outputs the string representing the instance of the class. 
- It is a virtual function, which must be overwritten. The `const` indicates that it doesn't a

`virtual std::string dtype() const { return "double"; }`
- outputs the type of the elements in the tensor
- not sure if this needs to be virtual 

`virtual ~BaseTensor() = default;` 
- a destructor. Not sure if this is needed

`const std::vector<size_t>& shape() const { return shape_; }` 
- getter function for the shape 

`const std::vector<double>& data() const { return storage_; }`
- getter function for the data, or storage

`BaseTensor& reshape(std::vector<size_t> new_shape);`
- simply reshapes by changing the `shape` attribute and does nothing else. 

`virtual bool operator==(BaseTensor& other) const;` 
- equality operator that minimally checks the attributes `storage_` and `shape_`. 
- Is a virtual function since it must be overwritten by `GradTensor`s which have additional `pivot` attributes. 

`virtual bool operator!=(BaseTensor& other) const;`
- Just negation of equality operator (see above). 

`operator std::string() const;`
- Returns a string so we can actually print tensors. Prints the type, plus the `storage_` and `shape_` so that we can see array structure. 

`virtual double at(const std::vector<size_t>& indices) const;`
- Similar to `__getitem__`, where you return a copy of an element by its index. 

`virtual double& at(const std::vector<size_t>& indices);`
- Similar to `__setitem__`, where you return a reference to the element for modification. 


`virtual std::unique_ptr<BaseTensor> slice(const std::vector<Slice>& slices) const;`
- Used to get a slice of a `Tensor` with the strides stored in `BaseTensor::Slice` struct. 
- Returns a copy, not a reference/view of the Tensor!

    ```
    struct Slice {
      size_t start;
      size_t stop;
      size_t step;
      
      Slice(size_t start_ = 0, 
        size_t stop_ = std::numeric_limits<size_t>::max(), 
        size_t step_ = 1)
      : start(start_), stop(stop_), step(step_) {}
    };
    ```

### Notes 

- I'm not sure whether to include strides as PyTorch does, as this is directly in the `shape`. This would certainly make viewing easier, but would require a lot of modification in the `std::string` function of `BaseTensor` to use the strides to push into a stringstream. 

- I've tried virtualizing the transpose function from `BaseTensor`, but I wanted it to return a reference for `GradTensor` while a copy for `Tensor`, so I made two separate implementations in both subclasses.  


## GradTensors 

Gradient Tensor, or `GradTensor`s, are tensors that store the total derivative of an elementary operation (not precisely the gradient, but in $\mathbb{R}^n$ one can be transposed to get the other). These operations can have 1 or more arguments. It is represented as an $N$-tensor of size 

$$
    (D_1, D_2, \ldots, D_N)
$$

Let's look at a regular gradient of a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, which is a $m \times n$ matrix. However, if we have another function $g: \mathbb{R}^{n \times m} \rightarrow \mathbb{R}$, then this also is a matrix of shape $m \times n$. Clearly there is some ambiguity here, so we must store another attribute which I call the *pivot dimension*, that captures this information. Say that we have $f: \mathbb{R}^{\mathbf{n}} \rightarrow \mathbb{R}^{\mathbf{m}}$, where the superscripts are now vectors of length $d_n, d_m$. Then, the total derivative has shape  
$$ 
    \mathbf{m} \times \mathbf{n}
$$
with the pivot being $d_m + 1$, the first dimension index of the input. 

Essentially, we are approaching matrix multiplication in a more general way by [contracting tensors](https://en.wikipedia.org/wiki/Tensor_contraction). Note that by including this pivot parameter, we can support both batching and higher-dimensional multiplication. 

### Attributes 

`std::vector<double> storage_`
- A contiguous vector  of doubles that store the state of the tensor. 

`std::vector<size_t> shape_`
- The shape of the tensor, where the product of the shape elements should match the length of `storage_`. 

`size_t pivot_`
- The pivot index that marks the start hyperdimension of the input. 

### Constructors 

`GradTensor();` 
- Default constructor that is called when initializing a tensor without any gradients. Stores empty vectors and `pivot_ = 0`.

`GradTensor(std::vector<double> data, std::vector<size_t> shape, size_t pivot);` 
- Full constructor that sets all attributes. 

`GradTensor(std::vector<size_t> shape, size_t pivot);` 
- Initializes a GradTensor of shape shape and pivot but with all $0$ entries. 

`static GradTensor eye(size_t n, size_t pivot = 1);` 
- Creates an identity matrix gradient, which is a good default initialization when calling `backprop()` on a tensor. 


### Methods

`std::string type() const override { return "GradTensor"; }`
- overrides type 

`size_t pivot() const { return pivot_; }`
- getter method for pivot index

`bool operator==(GradTensor& other) const;`
- equality operator that also checks equality in pivot 

`bool operator!=(GradTensor& other) const;`
- negation of equality. 

`GradTensor copy() const;`
- returns a new GradTensor copy

`GradTensor add(GradTensor& other);`
- For adding gradients together of the same shape and pivot. 

`Tensor add(Tensor& other);`
- For adding gradients to parameters for updating.  

`GradTensor sub(GradTensor& other);`
- For subtracting gradients together of the same shape and pivot. 

`Tensor sub(Tensor& other);`
- For subtracting gradients to parameters for updating.  

`GradTensor mul(GradTensor& other);`
- Elementwise multiplication, which doesn't really make sense to do at all but included.  
`Tensor mul(Tensor& other); `
- Elementwise multiplication, which doesn't really make sense to do at all but included.  
`GradTensor matmul(GradTensor& other); `
- Right matrix multiplication or tensor contraction, used for chain rule. 

`GradTensor& transpose(const std::vector<size_t>& axes = {});`
- Modifies the gradient tensor in place and returns itself. 
- It doesn't return a copy like in `Tensor` since I don't think it makes sense reuse the old one. If you must, you can just copy it and then transpose it. 

### Notes 

- The most natural operations on gradients/Jacobians are addition, subtraction, and multiplication (composition). Operations like taking the dot product or element-wise multiplication doesn't make sense, so I did not implement them on purpose. 

- Might need to check whether addition checks if pivots are the same. 

## Tensor

Regular tensors, or `Tensor`s, store either tabular data or the state of a parameter. 

### Attributes 

`std::vector<double> storage_`
- A contiguous vector  of doubles that store the state of the tensor. 

`std::vector<size_t> shape_`
- The shape of the tensor, where the product of the shape elements should match the length of `storage_`. 

`GradTensor grad = GradTensor();`
- the Jacobian (if being precise, rather than gradient) of some tensor further down the computation graph with respect to this tensor. 

`std::vector<Tensor*> prev = std::vector<Tensor*>();` 
- previous nodes used to compute this tensor, if any 

`std::function<void()> backward;`
- function for filling in gradients of this tensor

### Constructors 

`Tensor(std::vector<double> data, std::vector<size_t> shape);`
- The full constructor, which sets the storage and shape, whilst setting the gradients to null. 

`Tensor(std::vector<double> data);`
- Constructor for 1D arrays. 

`Tensor(std::vector<std::vector<double>> data);`
- Constructor for 2D arrays. 

`Tensor(std::vector<std::vector<std::vector<double>>> data);`
- Constructor for 3D arrays. 

`static Tensor arange(int start, int stop, int step = 1);`
- Arange constructor, returning a 1D array. 

`static Tensor linspace(double start, double stop, int numsteps);`
- Linspace constructor (like in numpy), returning a 1D array. 

`static Tensor gaussian(std::vector<size_t> shape, double mean = 0.0, double stddev = 1.0);`
- Returns a Tensor of shape `shape` of Gaussian random variables. 

`static Tensor uniform(std::vector<size_t> shape, double min = 0.0, double max = 1.0);`
- Returns a Tensor of shape `shape` of Uniform random variables. 

`static Tensor ones(std::vector<size_t> shape);`
- Returns a Tensor of shape `shape` of all $1$. 

`static Tensor zeros(std::vector<size_t> shape);`
- Returns a Tensor of shape `shape` of all $0$. 


### Methods 


