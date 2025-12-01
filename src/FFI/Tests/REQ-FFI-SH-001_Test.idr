-- Test for REQ-FFI-SH-001: at_dim binding for dimension count
module Main

import FFI.FFI

testTensorDim : IO ()
testTensorDim = do
  putStrLn "Testing at_dim binding..."
  ptr <- newTensor
  d <- tensorDim ptr
  putStrLn $ "  tensorDim returned: " ++ show d
  freeTensor ptr
  putStrLn "PASS: at_dim binding works"

main : IO ()
main = testTensorDim
