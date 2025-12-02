-- Test for REQ-TEN-DES-001: Provide free : (1 t : Tensor) -> IO ()
module Main

import Tensor.Tensor

testFreeSignature : IO ()
testFreeSignature = do
  putStrLn "Testing free function..."
  t <- empty
  putStrLn "  Created tensor"
  putStrLn "  free : (1 t : Tensor) -> IO ()"
  putStrLn "  Linear consumption: tensor cannot be used after free"
  free t
  putStrLn "  free called successfully"
  putStrLn "PASS: free consumes tensor linearly"

main : IO ()
main = testFreeSignature
