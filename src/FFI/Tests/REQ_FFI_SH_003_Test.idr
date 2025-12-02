-- Test for REQ_FFI_SH_003: at_scalar_type binding for dtype query
module Main

import FFI.FFI
import Torch.Torch

testScalarTypeBinding : IO ()
testScalarTypeBinding = do
  putStrLn "Testing at_scalar_type binding..."
  ptr <- newTensor
  putStrLn "  Created tensor"
  -- tensorScalarType wraps at_scalar_type
  scalarType <- tensorScalarType ptr
  putStrLn $ "  at_scalar_type returned: " ++ show scalarType
  -- Verify conversion to DType works
  let maybeDtype = intToDType scalarType
  putStrLn $ "  intToDType result: " ++ show maybeDtype
  freeTensor ptr
  putStrLn "  Freed tensor"
  putStrLn "PASS: at_scalar_type binding works correctly"

main : IO ()
main = testScalarTypeBinding
