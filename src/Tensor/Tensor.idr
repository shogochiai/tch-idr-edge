||| Linear Type Tensor Wrapper
||| Layer 2: The lazy-idris safety layer
||| Invariant: (1 t : Tensor) ensures exactly-once usage
module Tensor.Tensor

import FFI.FFI
import Torch.Torch

%default total

||| Opaque tensor type with linear constraint
||| This is the core type ensuring no implicit Drop
public export
data Tensor : Type where
  MkTensor : TensorPtr -> Tensor

||| Extract raw pointer (internal use only)
rawPtr : Tensor -> TensorPtr
rawPtr (MkTensor ptr) = ptr

-- ============================================================
-- Tensor Creation
-- ============================================================

||| Create an empty tensor
||| Returns: Linear tensor that MUST be freed
export
empty : IO Tensor
empty = do
  ptr <- newTensor
  pure (MkTensor ptr)

||| Create a 1D tensor of zeros
export
zeros1d : Bits64 -> IO Tensor
zeros1d size = do
  ptr <- tensorZeros1d size
  pure (MkTensor ptr)

||| Create a 1D tensor of ones
export
ones1d : Bits64 -> IO Tensor
ones1d size = do
  ptr <- tensorOnes1d size
  pure (MkTensor ptr)

||| Create a 2D tensor of zeros
export
zeros2d : Bits64 -> Bits64 -> IO Tensor
zeros2d d0 d1 = do
  ptr <- tensorZeros2d d0 d1
  pure (MkTensor ptr)

||| Create a 2D tensor of ones
export
ones2d : Bits64 -> Bits64 -> IO Tensor
ones2d d0 d1 = do
  ptr <- tensorOnes2d d0 d1
  pure (MkTensor ptr)

-- ============================================================
-- Tensor Inspection (Linear Preserving)
-- ============================================================

||| Print tensor to stdout
||| Linear: Returns the same tensor for continued use
export
printT : (1 t : Tensor) -> IO Tensor
printT (MkTensor ptr) = do
  printTensor ptr
  pure (MkTensor ptr)

||| Check if tensor is defined
||| Linear: Returns result AND the tensor
export
isDefined : (1 t : Tensor) -> IO (Bool, Tensor)
isDefined (MkTensor ptr) = do
  b <- tensorDefined ptr
  pure (b, MkTensor ptr)

||| Get tensor dimensionality
||| Linear: Returns result AND the tensor
export
dim : (1 t : Tensor) -> IO (Nat, Tensor)
dim (MkTensor ptr) = do
  d <- tensorDim ptr
  pure (cast d, MkTensor ptr)

-- ============================================================
-- Tensor Arithmetic (Linear - consumes inputs, produces output)
-- ============================================================

||| Element-wise addition
||| Linear: Consumes both inputs, produces new tensor
export
add : (1 a : Tensor) -> (1 b : Tensor) -> IO Tensor
add (MkTensor pa) (MkTensor pb) = do
  ptr <- tensorAdd pa pb
  pure (MkTensor ptr)

||| Element-wise multiplication
||| Linear: Consumes both inputs, produces new tensor
export
mul : (1 a : Tensor) -> (1 b : Tensor) -> IO Tensor
mul (MkTensor pa) (MkTensor pb) = do
  ptr <- tensorMul pa pb
  pure (MkTensor ptr)

||| Matrix multiplication
||| Linear: Consumes both inputs, produces new tensor
export
matmul : (1 a : Tensor) -> (1 b : Tensor) -> IO Tensor
matmul (MkTensor pa) (MkTensor pb) = do
  ptr <- tensorMatmul pa pb
  pure (MkTensor ptr)

-- ============================================================
-- Tensor Activations (Linear - consumes input, produces output)
-- ============================================================

||| Softmax along dimension
||| Linear: Consumes input, produces new tensor
export
softmax : (1 t : Tensor) -> Bits64 -> IO Tensor
softmax (MkTensor ptr) dim = do
  result <- tensorSoftmax ptr dim
  pure (MkTensor result)

||| ReLU activation
||| Linear: Consumes input, produces new tensor
export
relu : (1 t : Tensor) -> IO Tensor
relu (MkTensor ptr) = do
  result <- tensorRelu ptr
  pure (MkTensor result)

-- ============================================================
-- Tier 1: Shape Operations (Linear)
-- ============================================================

||| Transpose dimensions dim0 and dim1
||| Linear: Consumes input, produces new tensor
export
transpose : (1 t : Tensor) -> Bits64 -> Bits64 -> IO Tensor
transpose (MkTensor ptr) dim0 dim1 = do
  result <- tensorTranspose ptr dim0 dim1
  pure (MkTensor result)

||| Reshape to 2D
||| Linear: Consumes input, produces new tensor
export
reshape2d : (1 t : Tensor) -> Bits64 -> Bits64 -> IO Tensor
reshape2d (MkTensor ptr) d0 d1 = do
  result <- tensorReshape2d ptr d0 d1
  pure (MkTensor result)

||| Reshape to 3D
||| Linear: Consumes input, produces new tensor
export
reshape3d : (1 t : Tensor) -> Bits64 -> Bits64 -> Bits64 -> IO Tensor
reshape3d (MkTensor ptr) d0 d1 d2 = do
  result <- tensorReshape3d ptr d0 d1 d2
  pure (MkTensor result)

||| Reshape to 4D
||| Linear: Consumes input, produces new tensor
export
reshape4d : (1 t : Tensor) -> Bits64 -> Bits64 -> Bits64 -> Bits64 -> IO Tensor
reshape4d (MkTensor ptr) d0 d1 d2 d3 = do
  result <- tensorReshape4d ptr d0 d1 d2 d3
  pure (MkTensor result)

||| View as 2D (no copy, shares storage)
||| Linear: Consumes input, produces new tensor
export
view2d : (1 t : Tensor) -> Bits64 -> Bits64 -> IO Tensor
view2d (MkTensor ptr) d0 d1 = do
  result <- tensorView2d ptr d0 d1
  pure (MkTensor result)

||| View as 3D (no copy, shares storage)
||| Linear: Consumes input, produces new tensor
export
view3d : (1 t : Tensor) -> Bits64 -> Bits64 -> Bits64 -> IO Tensor
view3d (MkTensor ptr) d0 d1 d2 = do
  result <- tensorView3d ptr d0 d1 d2
  pure (MkTensor result)

||| View as 4D (no copy, shares storage)
||| Linear: Consumes input, produces new tensor
export
view4d : (1 t : Tensor) -> Bits64 -> Bits64 -> Bits64 -> Bits64 -> IO Tensor
view4d (MkTensor ptr) d0 d1 d2 d3 = do
  result <- tensorView4d ptr d0 d1 d2 d3
  pure (MkTensor result)

-- ============================================================
-- Tier 2: Tensor Creation (additional)
-- ============================================================

||| Create 1D random normal tensor
export
randn1d : Bits64 -> IO Tensor
randn1d d0 = do
  ptr <- tensorRandn1d d0
  pure (MkTensor ptr)

||| Create 2D random normal tensor
export
randn2d : Bits64 -> Bits64 -> IO Tensor
randn2d d0 d1 = do
  ptr <- tensorRandn2d d0 d1
  pure (MkTensor ptr)

||| Create 3D random normal tensor
export
randn3d : Bits64 -> Bits64 -> Bits64 -> IO Tensor
randn3d d0 d1 d2 = do
  ptr <- tensorRandn3d d0 d1 d2
  pure (MkTensor ptr)

-- ============================================================
-- Tier 3: Shape Queries (Linear Preserving)
-- ============================================================

||| Get size at dimension
||| Linear: Returns result AND the tensor
export
size : (1 t : Tensor) -> Bits64 -> IO (Bits64, Tensor)
size (MkTensor ptr) d = do
  s <- tensorSizeDim ptr d
  pure (s, MkTensor ptr)

||| Extract scalar value as Double
||| Linear: Returns result AND the tensor
export
item : (1 t : Tensor) -> IO (Double, Tensor)
item (MkTensor ptr) = do
  v <- tensorItemDouble ptr
  pure (v, MkTensor ptr)

-- ============================================================
-- Tier 4: Tensor Combination (Linear)
-- ============================================================

||| Concatenate 2 tensors along dimension
||| Linear: Consumes both inputs, produces new tensor
export
cat2 : (1 a : Tensor) -> (1 b : Tensor) -> Bits64 -> IO Tensor
cat2 (MkTensor pa) (MkTensor pb) dim = do
  result <- tensorCat2 pa pb dim
  pure (MkTensor result)

||| Stack 2 tensors along new dimension
||| Linear: Consumes both inputs, produces new tensor
export
stack2 : (1 a : Tensor) -> (1 b : Tensor) -> Bits64 -> IO Tensor
stack2 (MkTensor pa) (MkTensor pb) dim = do
  result <- tensorStack2 pa pb dim
  pure (MkTensor result)

-- ============================================================
-- Tier 5: Neural Network Primitives (Linear)
-- ============================================================

||| Layer normalization over last dimension
||| Linear: Consumes input, produces new tensor
export
layerNorm : (1 t : Tensor) -> Bits64 -> IO Tensor
layerNorm (MkTensor ptr) normDim = do
  result <- tensorLayerNorm1d ptr normDim 1.0e-5
  pure (MkTensor result)

||| Embedding lookup
||| Linear: Consumes indices, returns (result, weight) - weight is preserved
export
embedding : (1 weight : Tensor) -> (1 indices : Tensor) -> IO (Tensor, Tensor)
embedding (MkTensor wptr) (MkTensor iptr) = do
  result <- tensorEmbedding wptr iptr
  pure (MkTensor result, MkTensor wptr)

||| Dropout with probability p
||| Linear: Consumes input, produces new tensor
export
dropout : (1 t : Tensor) -> Double -> Bool -> IO Tensor
dropout (MkTensor ptr) p training = do
  result <- tensorDropout ptr p training
  pure (MkTensor result)

||| GELU activation
||| Linear: Consumes input, produces new tensor
export
gelu : (1 t : Tensor) -> IO Tensor
gelu (MkTensor ptr) = do
  result <- tensorGelu ptr
  pure (MkTensor result)

-- ============================================================
-- Tensor Ownership (Linear)
-- ============================================================

||| Duplicate tensor - splits ownership into two
||| Both returned tensors MUST be freed separately
||| Uses shallow clone (shares underlying storage)
||| Linear: Consumes input, produces two outputs
export
dup : (1 t : Tensor) -> IO (Tensor, Tensor)
dup (MkTensor ptr) = do
  ptr2 <- shallowClone ptr
  pure (MkTensor ptr, MkTensor ptr2)

-- ============================================================
-- Tensor Destruction (Linear Consuming)
-- ============================================================

||| Free tensor memory - MUST be called exactly once
||| Linear: Consumes the tensor, no further use allowed
export
free : (1 t : Tensor) -> IO ()
free (MkTensor ptr) = freeTensor ptr

-- ============================================================
-- Tier 5: Data Bridge (Linear)
-- ============================================================

||| Create tensor from List of Int64 values
||| Returns: Linear tensor that MUST be freed
export
fromListInt64 : List Bits64 -> IO Tensor
fromListInt64 xs = do
  ptr <- tensorFromListInt64 xs
  pure (MkTensor ptr)

||| Create tensor from List of Double values
||| Returns: Linear tensor that MUST be freed
export
fromListDouble : List Double -> IO Tensor
fromListDouble xs = do
  ptr <- tensorFromListDouble xs
  pure (MkTensor ptr)

-- ============================================================
-- Tier 5: Reduction Operations (Linear)
-- ============================================================

||| Mean along dimension
||| Linear: Consumes input, produces new tensor
export
mean : (1 t : Tensor) -> Bits64 -> IO Tensor
mean (MkTensor ptr) dim = do
  result <- tensorMeanDim ptr dim
  pure (MkTensor result)

||| Sum along dimension
||| Linear: Consumes input, produces new tensor
export
sum : (1 t : Tensor) -> Bits64 -> IO Tensor
sum (MkTensor ptr) dim = do
  result <- tensorSumDim ptr dim
  pure (MkTensor result)

-- ============================================================
-- Tier 5: Scalar Operations (Linear)
-- ============================================================

||| Divide tensor by scalar value
||| Linear: Consumes input, produces new tensor
export
divScalar : (1 t : Tensor) -> Double -> IO Tensor
divScalar (MkTensor ptr) val = do
  result <- tensorDivScalar ptr val
  pure (MkTensor result)

||| Multiply tensor by scalar value
||| Linear: Consumes input, produces new tensor
export
mulScalar : (1 t : Tensor) -> Double -> IO Tensor
mulScalar (MkTensor ptr) val = do
  result <- tensorMulScalar ptr val
  pure (MkTensor result)

-- ============================================================
-- Tier 7: Tensor Slicing (Linear)
-- ============================================================

||| Narrow tensor along dimension
||| Returns a view (shares underlying storage) with reduced size
||| Linear: Consumes input, produces narrowed tensor
||| @dim    Dimension to narrow along
||| @start  Starting index
||| @length Length of slice
export
narrow : (1 t : Tensor) -> Bits64 -> Bits64 -> Bits64 -> IO Tensor
narrow (MkTensor ptr) dim start length = do
  result <- tensorNarrow ptr dim start length
  pure (MkTensor result)

||| Split tensor into 3 equal parts along dimension 0
||| Useful for splitting combined Q/K/V projection weights
||| Linear: Consumes input, produces three tensors (all must be freed)
||| @splitSize Size of each split (total size must be 3*splitSize)
export
split3 : (1 t : Tensor) -> Bits64 -> IO (Tensor, Tensor, Tensor)
split3 (MkTensor ptr) splitSize = do
  -- Create clones so we can narrow multiple times
  ptr2 <- shallowClone ptr
  ptr3 <- shallowClone ptr
  -- Split: [0:splitSize], [splitSize:2*splitSize], [2*splitSize:3*splitSize]
  t1 <- tensorNarrow ptr 0 0 splitSize
  t2 <- tensorNarrow ptr2 0 splitSize splitSize
  t3 <- tensorNarrow ptr3 0 (splitSize * 2) splitSize
  pure (MkTensor t1, MkTensor t2, MkTensor t3)
