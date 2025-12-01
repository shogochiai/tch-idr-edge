||| Linear Type Tensor Wrapper
||| Layer 2: The lazy-idris safety layer
||| Invariant: (1 t : Tensor) ensures exactly-once usage
module Tensor.Tensor

import FFI.FFI
import Torch.Torch

%default total

||| Opaque tensor type with linear constraint
||| This is the core type ensuring no implicit Drop
export
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
-- Tensor Destruction (Linear Consuming)
-- ============================================================

||| Free tensor memory - MUST be called exactly once
||| Linear: Consumes the tensor, no further use allowed
export
free : (1 t : Tensor) -> IO ()
free (MkTensor ptr) = freeTensor ptr
