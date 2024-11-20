# Tensors 

The hierarchy is 
```
BaseTensor 
    Tensor 
        ScalarTensor 
        DenseTensor  (TBI)
        SparseTensor (TBI)
    GradTensor
``` 

## BaseTensor 

`BaseTensors` are abstract classes that support the very minimal functionalities that all tensors should have. This includes two attributes 
1. `storage`, a contiguous `std::vector` of data 
2. `shape`, which tells us how the tensor should be interpreted as 
3. I'm not sure whether to include strides as PyTorch does, as this is directly in the `shape`


