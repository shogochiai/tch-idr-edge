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
