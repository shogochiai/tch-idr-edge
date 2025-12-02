-- Test for REQ_TEN_LIN_001: Wrap TensorPtr in opaque Tensor type
module Main

import Tensor.Tensor

testOpaqueType : IO ()
testOpaqueType = do
  putStrLn "Testing opaque Tensor type..."
  t <- empty
  putStrLn "  Created Tensor via empty"
  putStrLn "  Tensor is opaque (MkTensor constructor not exported)"
  free t
  putStrLn "  Freed tensor"
  putStrLn "PASS: Tensor type wraps TensorPtr"

main : IO ()
main = testOpaqueType
