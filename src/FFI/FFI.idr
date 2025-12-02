||| Raw FFI bindings to LibTorch C++ shim
||| Layer 1: Unsafe PrimIO bindings - no safety guarantees
module FFI.FFI

import System.FFI

%default total

-- Raw tensor pointer type (opaque)
public export
TensorPtr : Type
TensorPtr = AnyPtr

-- FFI library specification
%foreign "C:get_and_reset_last_err,libtorch_shim"
prim__getLastErr : PrimIO AnyPtr

%foreign "C:at_new_tensor,libtorch_shim"
prim__newTensor : PrimIO TensorPtr

%foreign "C:at_tensor_of_data,libtorch_shim"
prim__tensorOfData : AnyPtr -> AnyPtr -> Bits64 -> Bits64 -> Int -> PrimIO TensorPtr

%foreign "C:at_shallow_clone,libtorch_shim"
prim__shallowClone : TensorPtr -> PrimIO TensorPtr

%foreign "C:at_free,libtorch_shim"
prim__free : TensorPtr -> PrimIO ()

%foreign "C:at_print,libtorch_shim"
prim__print : TensorPtr -> PrimIO ()

%foreign "C:at_dim,libtorch_shim"
prim__dim : TensorPtr -> PrimIO Bits64

%foreign "C:at_scalar_type,libtorch_shim"
prim__scalarType : TensorPtr -> PrimIO Int

%foreign "C:at_device,libtorch_shim"
prim__device : TensorPtr -> PrimIO Int

%foreign "C:at_defined,libtorch_shim"
prim__defined : TensorPtr -> PrimIO Int

%foreign "C:at_data_ptr,libtorch_shim"
prim__dataPtr : TensorPtr -> PrimIO AnyPtr

-- ============================================================
-- Arithmetic Operations (out-parameter pattern)
-- ============================================================

%foreign "C:atg_add,libtorch_shim"
prim__add : AnyPtr -> TensorPtr -> TensorPtr -> PrimIO ()

%foreign "C:atg_mul,libtorch_shim"
prim__mul : AnyPtr -> TensorPtr -> TensorPtr -> PrimIO ()

%foreign "C:atg_matmul,libtorch_shim"
prim__matmul : AnyPtr -> TensorPtr -> TensorPtr -> PrimIO ()

-- Helper for reading out-parameter results
%foreign "C:idris_read_tensor_ptr,libtorch_shim"
prim__readOutPtr : AnyPtr -> PrimIO TensorPtr

-- ============================================================
-- Tensor Creation Operations (simplified helpers)
-- ============================================================

%foreign "C:idris_zeros_1d,libtorch_shim"
prim__zeros1d : AnyPtr -> Bits64 -> Int -> Int -> PrimIO ()

%foreign "C:idris_ones_1d,libtorch_shim"
prim__ones1d : AnyPtr -> Bits64 -> Int -> Int -> PrimIO ()

%foreign "C:idris_zeros_2d,libtorch_shim"
prim__zeros2d : AnyPtr -> Bits64 -> Bits64 -> Int -> Int -> PrimIO ()

%foreign "C:idris_ones_2d,libtorch_shim"
prim__ones2d : AnyPtr -> Bits64 -> Bits64 -> Int -> Int -> PrimIO ()

-- ============================================================
-- Activation Operations
-- ============================================================

%foreign "C:atg_softmax,libtorch_shim"
prim__softmax : AnyPtr -> TensorPtr -> Bits64 -> Int -> PrimIO ()

%foreign "C:atg_relu,libtorch_shim"
prim__relu : AnyPtr -> TensorPtr -> PrimIO ()

-- ============================================================
-- Debug helpers (prim bindings)
-- ============================================================

%foreign "C:idris_debug_echo,libtorch_shim"
prim__debugEcho : Bits64 -> PrimIO Bits64

%foreign "C:idris_debug_outptr,libtorch_shim"
prim__debugOutptr : AnyPtr -> Bits64 -> PrimIO ()

-- ============================================================
-- Tier 1: Shape Operations (prim bindings)
-- ============================================================

%foreign "C:idris_transpose,libtorch_shim"
prim__transpose : AnyPtr -> TensorPtr -> Bits64 -> Bits64 -> PrimIO ()

%foreign "C:idris_reshape_1d,libtorch_shim"
prim__reshape1d : AnyPtr -> TensorPtr -> Bits64 -> PrimIO ()

%foreign "C:idris_reshape_2d,libtorch_shim"
prim__reshape2d : AnyPtr -> TensorPtr -> Bits64 -> Bits64 -> PrimIO ()

%foreign "C:idris_reshape_3d,libtorch_shim"
prim__reshape3d : AnyPtr -> TensorPtr -> Bits64 -> Bits64 -> Bits64 -> PrimIO ()

%foreign "C:idris_reshape_4d,libtorch_shim"
prim__reshape4d : AnyPtr -> TensorPtr -> Bits64 -> Bits64 -> Bits64 -> Bits64 -> PrimIO ()

%foreign "C:idris_view_1d,libtorch_shim"
prim__view1d : AnyPtr -> TensorPtr -> Bits64 -> PrimIO ()

%foreign "C:idris_view_2d,libtorch_shim"
prim__view2d : AnyPtr -> TensorPtr -> Bits64 -> Bits64 -> PrimIO ()

%foreign "C:idris_view_3d,libtorch_shim"
prim__view3d : AnyPtr -> TensorPtr -> Bits64 -> Bits64 -> Bits64 -> PrimIO ()

%foreign "C:idris_view_4d,libtorch_shim"
prim__view4d : AnyPtr -> TensorPtr -> Bits64 -> Bits64 -> Bits64 -> Bits64 -> PrimIO ()

-- ============================================================
-- Tier 2: Tensor Creation (prim bindings)
-- ============================================================

%foreign "C:idris_randn_1d,libtorch_shim"
prim__randn1d : AnyPtr -> Bits64 -> Int -> Int -> PrimIO ()

%foreign "C:idris_randn_2d,libtorch_shim"
prim__randn2d : AnyPtr -> Bits64 -> Bits64 -> Int -> Int -> PrimIO ()

%foreign "C:idris_randn_3d,libtorch_shim"
prim__randn3d : AnyPtr -> Bits64 -> Bits64 -> Bits64 -> Int -> Int -> PrimIO ()

-- ============================================================
-- Tier 3: Shape Queries (prim bindings)
-- ============================================================

%foreign "C:idris_size_dim,libtorch_shim"
prim__sizeDim : TensorPtr -> Bits64 -> PrimIO Bits64

%foreign "C:idris_item_double,libtorch_shim"
prim__itemDouble : TensorPtr -> PrimIO Double

-- ============================================================
-- Tier 4: Tensor Combination (prim bindings)
-- ============================================================

%foreign "C:idris_cat_2,libtorch_shim"
prim__cat2 : AnyPtr -> TensorPtr -> TensorPtr -> Bits64 -> PrimIO ()

%foreign "C:idris_cat_3,libtorch_shim"
prim__cat3 : AnyPtr -> TensorPtr -> TensorPtr -> TensorPtr -> Bits64 -> PrimIO ()

%foreign "C:idris_stack_2,libtorch_shim"
prim__stack2 : AnyPtr -> TensorPtr -> TensorPtr -> Bits64 -> PrimIO ()

%foreign "C:idris_stack_3,libtorch_shim"
prim__stack3 : AnyPtr -> TensorPtr -> TensorPtr -> TensorPtr -> Bits64 -> PrimIO ()

-- ============================================================
-- Tier 5: Neural Network Primitives (prim bindings)
-- ============================================================

%foreign "C:idris_layer_norm_1d,libtorch_shim"
prim__layerNorm1d : AnyPtr -> TensorPtr -> Bits64 -> Double -> PrimIO ()

%foreign "C:idris_layer_norm_2d,libtorch_shim"
prim__layerNorm2d : AnyPtr -> TensorPtr -> Bits64 -> Bits64 -> Double -> PrimIO ()

%foreign "C:idris_embedding,libtorch_shim"
prim__embedding : AnyPtr -> TensorPtr -> TensorPtr -> PrimIO ()

%foreign "C:idris_dropout,libtorch_shim"
prim__dropout : AnyPtr -> TensorPtr -> Double -> Int -> PrimIO ()

%foreign "C:idris_gelu,libtorch_shim"
prim__gelu : AnyPtr -> TensorPtr -> PrimIO ()

-- ============================================================
-- Tier 5: Data Bridge (prim bindings)
-- ============================================================

%foreign "C:idris_from_array_int64,libtorch_shim"
prim__fromArrayInt64 : AnyPtr -> AnyPtr -> Bits64 -> PrimIO ()

%foreign "C:idris_from_array_double,libtorch_shim"
prim__fromArrayDouble : AnyPtr -> AnyPtr -> Bits64 -> PrimIO ()

%foreign "C:idris_write_int64,libtorch_shim"
prim__writeInt64 : AnyPtr -> Bits64 -> Bits64 -> PrimIO ()

%foreign "C:idris_write_double,libtorch_shim"
prim__writeDouble : AnyPtr -> Bits64 -> Double -> PrimIO ()

-- ============================================================
-- Tier 5: Reduction Operations (prim bindings)
-- ============================================================

%foreign "C:idris_mean_dim,libtorch_shim"
prim__meanDim : AnyPtr -> TensorPtr -> Bits64 -> Int -> PrimIO ()

%foreign "C:idris_sum_dim,libtorch_shim"
prim__sumDim : AnyPtr -> TensorPtr -> Bits64 -> Int -> PrimIO ()

-- ============================================================
-- Tier 5: Scalar Operations (prim bindings)
-- ============================================================

%foreign "C:idris_div_scalar,libtorch_shim"
prim__divScalar : AnyPtr -> TensorPtr -> Double -> PrimIO ()

%foreign "C:idris_mul_scalar,libtorch_shim"
prim__mulScalar : AnyPtr -> TensorPtr -> Double -> PrimIO ()

-- Wrapped IO functions (still unsafe, but IO-typed)
export
getLastErr : IO (Maybe String)
getLastErr = do
  ptr <- primIO prim__getLastErr
  if prim__nullAnyPtr ptr /= 0
     then pure Nothing
     else pure (Just "Error occurred")  -- TODO: proper string conversion

export
newTensor : IO TensorPtr
newTensor = primIO prim__newTensor

export
freeTensor : TensorPtr -> IO ()
freeTensor ptr = primIO (prim__free ptr)

export
printTensor : TensorPtr -> IO ()
printTensor ptr = primIO (prim__print ptr)

export
tensorDim : TensorPtr -> IO Bits64
tensorDim ptr = primIO (prim__dim ptr)

export
tensorDefined : TensorPtr -> IO Bool
tensorDefined ptr = do
  r <- primIO (prim__defined ptr)
  pure (r /= 0)

export
tensorScalarType : TensorPtr -> IO Int
tensorScalarType ptr = primIO (prim__scalarType ptr)

export
tensorDevice : TensorPtr -> IO Int
tensorDevice ptr = primIO (prim__device ptr)

export
tensorDataPtr : TensorPtr -> IO AnyPtr
tensorDataPtr ptr = primIO (prim__dataPtr ptr)

export
shallowClone : TensorPtr -> IO TensorPtr
shallowClone ptr = primIO (prim__shallowClone ptr)

-- ============================================================
-- Memory allocation for out-parameters
-- ============================================================

%foreign "C:malloc,libc 6"
prim__malloc : Bits64 -> PrimIO AnyPtr

%foreign "C:free,libc 6"
prim__freePtr : AnyPtr -> PrimIO ()

||| Allocate space for a single pointer (out-parameter)
allocOutPtr : IO AnyPtr
allocOutPtr = primIO (prim__malloc 8)  -- sizeof(void*)

||| Free out-parameter allocation
freeOutPtr : AnyPtr -> IO ()
freeOutPtr ptr = primIO (prim__freePtr ptr)

||| Read tensor result from out-parameter
readOutPtr : AnyPtr -> IO TensorPtr
readOutPtr ptr = primIO (prim__readOutPtr ptr)

-- ============================================================
-- Arithmetic Operations (wrapped)
-- ============================================================

||| Element-wise addition: a + b
export
tensorAdd : TensorPtr -> TensorPtr -> IO TensorPtr
tensorAdd a b = do
  out <- allocOutPtr
  primIO (prim__add out a b)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Element-wise multiplication: a * b
export
tensorMul : TensorPtr -> TensorPtr -> IO TensorPtr
tensorMul a b = do
  out <- allocOutPtr
  primIO (prim__mul out a b)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Matrix multiplication: a @ b
export
tensorMatmul : TensorPtr -> TensorPtr -> IO TensorPtr
tensorMatmul a b = do
  out <- allocOutPtr
  primIO (prim__matmul out a b)
  result <- readOutPtr out
  freeOutPtr out
  pure result

-- ============================================================
-- Activation Operations (wrapped)
-- ============================================================

||| Softmax along dimension
||| dtype: -1 for same as input, or scalar type code
export
tensorSoftmax : TensorPtr -> Bits64 -> IO TensorPtr
tensorSoftmax t dim = do
  out <- allocOutPtr
  primIO (prim__softmax out t dim (-1))  -- -1 = keep dtype
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| ReLU activation
export
tensorRelu : TensorPtr -> IO TensorPtr
tensorRelu t = do
  out <- allocOutPtr
  primIO (prim__relu out t)
  result <- readOutPtr out
  freeOutPtr out
  pure result

-- ============================================================
-- Tensor Creation (wrapped)
-- ============================================================

||| Create 1D zeros tensor
||| dtype: 6 = Float32, device: -1 = CPU (negative values = CPU)
export
tensorZeros1d : Bits64 -> IO TensorPtr
tensorZeros1d size = do
  out <- allocOutPtr
  primIO (prim__zeros1d out size 6 (-1))  -- Float32, CPU
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Create 1D ones tensor
export
tensorOnes1d : Bits64 -> IO TensorPtr
tensorOnes1d size = do
  out <- allocOutPtr
  primIO (prim__ones1d out size 6 (-1))  -- Float32, CPU
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Create 2D zeros tensor
export
tensorZeros2d : Bits64 -> Bits64 -> IO TensorPtr
tensorZeros2d d0 d1 = do
  out <- allocOutPtr
  primIO (prim__zeros2d out d0 d1 6 (-1))  -- Float32, CPU
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Create 2D ones tensor
export
tensorOnes2d : Bits64 -> Bits64 -> IO TensorPtr
tensorOnes2d d0 d1 = do
  out <- allocOutPtr
  primIO (prim__ones2d out d0 d1 6 (-1))  -- Float32, CPU
  result <- readOutPtr out
  freeOutPtr out
  pure result

-- ============================================================
-- Debug helpers (wrapped)
-- ============================================================

export
debugEcho : Bits64 -> IO Bits64
debugEcho x = primIO (prim__debugEcho x)

export
debugOutptr : Bits64 -> IO AnyPtr
debugOutptr val = do
  out <- allocOutPtr
  primIO (prim__debugOutptr out val)
  result <- readOutPtr out
  freeOutPtr out
  pure result

-- ============================================================
-- Tier 1: Shape Operations (wrapped)
-- ============================================================

||| Transpose dimensions dim0 and dim1
export
tensorTranspose : TensorPtr -> Bits64 -> Bits64 -> IO TensorPtr
tensorTranspose t dim0 dim1 = do
  out <- allocOutPtr
  primIO (prim__transpose out t dim0 dim1)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Reshape to 1D
export
tensorReshape1d : TensorPtr -> Bits64 -> IO TensorPtr
tensorReshape1d t d0 = do
  out <- allocOutPtr
  primIO (prim__reshape1d out t d0)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Reshape to 2D
export
tensorReshape2d : TensorPtr -> Bits64 -> Bits64 -> IO TensorPtr
tensorReshape2d t d0 d1 = do
  out <- allocOutPtr
  primIO (prim__reshape2d out t d0 d1)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Reshape to 3D
export
tensorReshape3d : TensorPtr -> Bits64 -> Bits64 -> Bits64 -> IO TensorPtr
tensorReshape3d t d0 d1 d2 = do
  out <- allocOutPtr
  primIO (prim__reshape3d out t d0 d1 d2)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Reshape to 4D
export
tensorReshape4d : TensorPtr -> Bits64 -> Bits64 -> Bits64 -> Bits64 -> IO TensorPtr
tensorReshape4d t d0 d1 d2 d3 = do
  out <- allocOutPtr
  primIO (prim__reshape4d out t d0 d1 d2 d3)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| View as 1D (no copy)
export
tensorView1d : TensorPtr -> Bits64 -> IO TensorPtr
tensorView1d t d0 = do
  out <- allocOutPtr
  primIO (prim__view1d out t d0)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| View as 2D (no copy)
export
tensorView2d : TensorPtr -> Bits64 -> Bits64 -> IO TensorPtr
tensorView2d t d0 d1 = do
  out <- allocOutPtr
  primIO (prim__view2d out t d0 d1)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| View as 3D (no copy)
export
tensorView3d : TensorPtr -> Bits64 -> Bits64 -> Bits64 -> IO TensorPtr
tensorView3d t d0 d1 d2 = do
  out <- allocOutPtr
  primIO (prim__view3d out t d0 d1 d2)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| View as 4D (no copy)
export
tensorView4d : TensorPtr -> Bits64 -> Bits64 -> Bits64 -> Bits64 -> IO TensorPtr
tensorView4d t d0 d1 d2 d3 = do
  out <- allocOutPtr
  primIO (prim__view4d out t d0 d1 d2 d3)
  result <- readOutPtr out
  freeOutPtr out
  pure result

-- ============================================================
-- Tier 2: Tensor Creation (wrapped)
-- ============================================================

||| Create 1D random normal tensor
export
tensorRandn1d : Bits64 -> IO TensorPtr
tensorRandn1d d0 = do
  out <- allocOutPtr
  primIO (prim__randn1d out d0 6 (-1))  -- Float32, CPU
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Create 2D random normal tensor
export
tensorRandn2d : Bits64 -> Bits64 -> IO TensorPtr
tensorRandn2d d0 d1 = do
  out <- allocOutPtr
  primIO (prim__randn2d out d0 d1 6 (-1))  -- Float32, CPU
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Create 3D random normal tensor
export
tensorRandn3d : Bits64 -> Bits64 -> Bits64 -> IO TensorPtr
tensorRandn3d d0 d1 d2 = do
  out <- allocOutPtr
  primIO (prim__randn3d out d0 d1 d2 6 (-1))  -- Float32, CPU
  result <- readOutPtr out
  freeOutPtr out
  pure result

-- ============================================================
-- Tier 3: Shape Queries (wrapped)
-- ============================================================

||| Get size at dimension
export
tensorSizeDim : TensorPtr -> Bits64 -> IO Bits64
tensorSizeDim t dim = primIO (prim__sizeDim t dim)

||| Extract scalar value as Double
export
tensorItemDouble : TensorPtr -> IO Double
tensorItemDouble t = primIO (prim__itemDouble t)

-- ============================================================
-- Tier 4: Tensor Combination (wrapped)
-- ============================================================

||| Concatenate 2 tensors along dimension
export
tensorCat2 : TensorPtr -> TensorPtr -> Bits64 -> IO TensorPtr
tensorCat2 t0 t1 dim = do
  out <- allocOutPtr
  primIO (prim__cat2 out t0 t1 dim)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Concatenate 3 tensors along dimension
export
tensorCat3 : TensorPtr -> TensorPtr -> TensorPtr -> Bits64 -> IO TensorPtr
tensorCat3 t0 t1 t2 dim = do
  out <- allocOutPtr
  primIO (prim__cat3 out t0 t1 t2 dim)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Stack 2 tensors along new dimension
export
tensorStack2 : TensorPtr -> TensorPtr -> Bits64 -> IO TensorPtr
tensorStack2 t0 t1 dim = do
  out <- allocOutPtr
  primIO (prim__stack2 out t0 t1 dim)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Stack 3 tensors along new dimension
export
tensorStack3 : TensorPtr -> TensorPtr -> TensorPtr -> Bits64 -> IO TensorPtr
tensorStack3 t0 t1 t2 dim = do
  out <- allocOutPtr
  primIO (prim__stack3 out t0 t1 t2 dim)
  result <- readOutPtr out
  freeOutPtr out
  pure result

-- ============================================================
-- Tier 5: Neural Network Primitives (wrapped)
-- ============================================================

||| Layer normalization over last dimension
export
tensorLayerNorm1d : TensorPtr -> Bits64 -> Double -> IO TensorPtr
tensorLayerNorm1d t normDim eps = do
  out <- allocOutPtr
  primIO (prim__layerNorm1d out t normDim eps)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Layer normalization over last 2 dimensions
export
tensorLayerNorm2d : TensorPtr -> Bits64 -> Bits64 -> Double -> IO TensorPtr
tensorLayerNorm2d t d0 d1 eps = do
  out <- allocOutPtr
  primIO (prim__layerNorm2d out t d0 d1 eps)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Embedding lookup
export
tensorEmbedding : TensorPtr -> TensorPtr -> IO TensorPtr
tensorEmbedding weight indices = do
  out <- allocOutPtr
  primIO (prim__embedding out weight indices)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Dropout
export
tensorDropout : TensorPtr -> Double -> Bool -> IO TensorPtr
tensorDropout t p training = do
  out <- allocOutPtr
  primIO (prim__dropout out t p (if training then 1 else 0))
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| GELU activation
export
tensorGelu : TensorPtr -> IO TensorPtr
tensorGelu t = do
  out <- allocOutPtr
  primIO (prim__gelu out t)
  result <- readOutPtr out
  freeOutPtr out
  pure result

-- ============================================================
-- Tier 5: Data Bridge (wrapped)
-- ============================================================

-- Buffer allocation for List marshalling
%foreign "C:calloc,libc 6"
prim__calloc : Bits64 -> Bits64 -> PrimIO AnyPtr

||| Allocate zeroed buffer
export
allocBuffer : Bits64 -> Bits64 -> IO AnyPtr
allocBuffer count size = primIO (prim__calloc count size)

||| Create tensor from Int64 list
export
tensorFromListInt64 : List Bits64 -> IO TensorPtr
tensorFromListInt64 xs = do
  let len = cast {to=Bits64} (length xs)
  buf <- allocBuffer len 8  -- 8 bytes per Int64
  writeLoop buf 0 xs
  out <- allocOutPtr
  primIO (prim__fromArrayInt64 out buf len)
  result <- readOutPtr out
  freeOutPtr out
  primIO (prim__freePtr buf)
  pure result
  where
    writeLoop : AnyPtr -> Bits64 -> List Bits64 -> IO ()
    writeLoop _ _ [] = pure ()
    writeLoop buf idx (v :: vs) = do
      primIO (prim__writeInt64 buf idx v)
      writeLoop buf (idx + 1) vs

||| Create tensor from Double list
export
tensorFromListDouble : List Double -> IO TensorPtr
tensorFromListDouble xs = do
  let len = cast {to=Bits64} (length xs)
  buf <- allocBuffer len 8  -- 8 bytes per Double
  writeLoop buf 0 xs
  out <- allocOutPtr
  primIO (prim__fromArrayDouble out buf len)
  result <- readOutPtr out
  freeOutPtr out
  primIO (prim__freePtr buf)
  pure result
  where
    writeLoop : AnyPtr -> Bits64 -> List Double -> IO ()
    writeLoop _ _ [] = pure ()
    writeLoop buf idx (v :: vs) = do
      primIO (prim__writeDouble buf idx v)
      writeLoop buf (idx + 1) vs

-- ============================================================
-- Tier 5: Reduction Operations (wrapped)
-- ============================================================

||| Mean along dimension
export
tensorMeanDim : TensorPtr -> Bits64 -> IO TensorPtr
tensorMeanDim t dim = do
  out <- allocOutPtr
  primIO (prim__meanDim out t dim 0)  -- keepdim=0
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Sum along dimension
export
tensorSumDim : TensorPtr -> Bits64 -> IO TensorPtr
tensorSumDim t dim = do
  out <- allocOutPtr
  primIO (prim__sumDim out t dim 0)  -- keepdim=0
  result <- readOutPtr out
  freeOutPtr out
  pure result

-- ============================================================
-- Tier 5: Scalar Operations (wrapped)
-- ============================================================

||| Divide tensor by scalar
export
tensorDivScalar : TensorPtr -> Double -> IO TensorPtr
tensorDivScalar t val = do
  out <- allocOutPtr
  primIO (prim__divScalar out t val)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Multiply tensor by scalar
export
tensorMulScalar : TensorPtr -> Double -> IO TensorPtr
tensorMulScalar t val = do
  out <- allocOutPtr
  primIO (prim__mulScalar out t val)
  result <- readOutPtr out
  freeOutPtr out
  pure result

-- ============================================================
-- Tier 6: StateDict Loading (prim bindings)
-- ============================================================

-- Raw state dict pointer type (opaque)
public export
StateDictPtr : Type
StateDictPtr = AnyPtr

%foreign "C:idris_load_state_dict,libtorch_shim"
prim__loadStateDict : String -> PrimIO StateDictPtr

%foreign "C:idris_state_dict_size,libtorch_shim"
prim__stateDictSize : StateDictPtr -> PrimIO Bits64

%foreign "C:idris_state_dict_error,libtorch_shim"
prim__stateDictError : StateDictPtr -> PrimIO AnyPtr

%foreign "C:idris_state_dict_name_at,libtorch_shim"
prim__stateDictNameAt : StateDictPtr -> Bits64 -> PrimIO AnyPtr

%foreign "C:idris_state_dict_tensor_at,libtorch_shim"
prim__stateDictTensorAt : AnyPtr -> StateDictPtr -> Bits64 -> PrimIO ()

%foreign "C:idris_state_dict_tensor_by_name,libtorch_shim"
prim__stateDictTensorByName : AnyPtr -> StateDictPtr -> String -> PrimIO ()

%foreign "C:idris_state_dict_free,libtorch_shim"
prim__stateDictFree : StateDictPtr -> PrimIO ()

-- ============================================================
-- Tier 6: StateDict Loading (wrapped)
-- ============================================================

||| Load state dict from checkpoint file
||| Returns handle to loaded state dict (may have error, check with stateDictError)
export
loadStateDictRaw : String -> IO StateDictPtr
loadStateDictRaw path = primIO (prim__loadStateDict path)

||| Get number of tensors in state dict
export
stateDictSize : StateDictPtr -> IO Bits64
stateDictSize sd = primIO (prim__stateDictSize sd)

-- Helper to cast C string pointer to Idris String
-- This is a simplified version - proper implementation would use strlen + memcpy
%foreign "C:strlen,libc 6"
prim__strlen : AnyPtr -> PrimIO Bits64

prim__castPtr : AnyPtr -> String
prim__castPtr ptr = believe_me ptr

||| Check for error message (returns Nothing if no error)
export
stateDictError : StateDictPtr -> IO (Maybe String)
stateDictError sd = do
  errPtr <- primIO (prim__stateDictError sd)
  if prim__nullAnyPtr errPtr /= 0
     then pure Nothing
     else pure (Just (prim__castPtr errPtr))

||| Get tensor name at index (returns Nothing if index out of bounds)
export
stateDictNameAt : StateDictPtr -> Bits64 -> IO (Maybe String)
stateDictNameAt sd idx = do
  namePtr <- primIO (prim__stateDictNameAt sd idx)
  if prim__nullAnyPtr namePtr /= 0
     then pure Nothing
     else pure (Just $ prim__castPtr namePtr)

||| Get tensor at index (shallow clone, caller owns result)
export
stateDictTensorAt : StateDictPtr -> Bits64 -> IO TensorPtr
stateDictTensorAt sd idx = do
  out <- allocOutPtr
  primIO (prim__stateDictTensorAt out sd idx)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Get tensor by name (shallow clone, caller owns result)
export
stateDictTensorByName : StateDictPtr -> String -> IO TensorPtr
stateDictTensorByName sd name = do
  out <- allocOutPtr
  primIO (prim__stateDictTensorByName out sd name)
  result <- readOutPtr out
  freeOutPtr out
  pure result

||| Free state dict handle and all contained tensors
export
freeStateDictRaw : StateDictPtr -> IO ()
freeStateDictRaw sd = primIO (prim__stateDictFree sd)

-- ============================================================
-- Tier 7: Tensor Slicing (prim bindings)
-- ============================================================

-- Use atg_narrow directly from libtorch (available in existing shim)
%foreign "C:atg_narrow,libtorch_shim"
prim__narrow : AnyPtr -> TensorPtr -> Bits64 -> Bits64 -> Bits64 -> PrimIO ()

-- ============================================================
-- Tier 7: Tensor Slicing (wrapped)
-- ============================================================

||| Narrow tensor along dimension
||| Returns a view with reduced size along specified dimension
||| @t      Input tensor
||| @dim    Dimension to narrow
||| @start  Starting index
||| @length Length of the narrowed dimension
export
tensorNarrow : TensorPtr -> Bits64 -> Bits64 -> Bits64 -> IO TensorPtr
tensorNarrow t dim start length = do
  out <- allocOutPtr
  primIO (prim__narrow out t dim start length)
  result <- readOutPtr out
  freeOutPtr out
  pure result
