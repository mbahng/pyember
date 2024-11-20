# GradTensors 

Gradient Tensor, or `GradTensor`s, are tensors that store the total derivative of an elementary operation (not precisely the gradient, but in $\mathbb{R}^n$ one can be transposed to get the other). These operations can have 1 or more arguments. It is represented as an $N$-tensor of size 

$$
    (D_1, D_2, \ldots, D_N)
$$

Let's look at a regular gradient of a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, which is a $m \times n$ matrix. However, if we have another function $g: \mathbb{R}^{n \times m} \rightarrow \mathbb{R}$, then this also is a matrix of shape $m \times n$. Clearly there is some ambiguity here, so we must store another attribute which I call the *pivot dimension*, that captures this information. Say that we have $f: \mathbb{R}^{\mathbf{n}} \rightarrow \mathbb{R}^{\mathbf{m}}$, where the superscripts are now vectors of length $d_n, d_m$. Then, the total derivative has shape  
$$ 
    \mathbf{m} \times \mathbf{n}
$$
with the pivot being $d_m + 1$, the first dimension index of the input. 

Essentially, we are approaching matrix multiplication in a more general way by [contracting tensors](https://en.wikipedia.org/wiki/Tensor_contraction). 
