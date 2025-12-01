||| Common types for LibTorch bindings
||| Layer: Types foundation for FFI and Tensor modules
module Torch.Torch

import Data.List

%default total

||| Tensor element types matching torch::ScalarType
public export
data DType = TFloat32 | TFloat64 | TInt32 | TInt64 | TInt16 | TInt8 | TUInt8 | TBool

public export
Show DType where
  show TFloat32 = "TFloat32"
  show TFloat64 = "TFloat64"
  show TInt32   = "TInt32"
  show TInt64   = "TInt64"
  show TInt16   = "TInt16"
  show TInt8    = "TInt8"
  show TUInt8   = "TUInt8"
  show TBool    = "TBool"

public export
Eq DType where
  TFloat32 == TFloat32 = True
  TFloat64 == TFloat64 = True
  TInt32 == TInt32 = True
  TInt64 == TInt64 = True
  TInt16 == TInt16 = True
  TInt8 == TInt8 = True
  TUInt8 == TUInt8 = True
  TBool == TBool = True
  _ == _ = False

||| Convert DType to LibTorch scalar type integer
public export
dtypeToInt : DType -> Int
dtypeToInt TFloat32 = 6
dtypeToInt TFloat64 = 7
dtypeToInt TInt32   = 3
dtypeToInt TInt64   = 4
dtypeToInt TInt16   = 2
dtypeToInt TInt8    = 1
dtypeToInt TUInt8   = 0
dtypeToInt TBool    = 11

||| Convert LibTorch scalar type integer to DType
public export
intToDType : Int -> Maybe DType
intToDType 0  = Just TUInt8
intToDType 1  = Just TInt8
intToDType 2  = Just TInt16
intToDType 3  = Just TInt32
intToDType 4  = Just TInt64
intToDType 6  = Just TFloat32
intToDType 7  = Just TFloat64
intToDType 11 = Just TBool
intToDType _  = Nothing

||| Computation device
public export
data Device = CPU | CUDA Nat

||| Convert Device to integer for FFI
public export
deviceToInt : Device -> Int
deviceToInt CPU      = -1
deviceToInt (CUDA n) = cast n

||| Tensor shape as list of dimensions
public export
Shape : Type
Shape = List Int

||| Element size in bytes for each dtype
public export
elementSize : DType -> Nat
elementSize TFloat32 = 4
elementSize TFloat64 = 8
elementSize TInt32   = 4
elementSize TInt64   = 8
elementSize TInt16   = 2
elementSize TInt8    = 1
elementSize TUInt8   = 1
elementSize TBool    = 1
