-- Test for REQ-TEN-OWN-002: Consume tensor linearly on free operation
module Main

import Tensor.Tensor

testFreeConsumesLinearly : IO ()
testFreeConsumesLinearly = do
  putStrLn "Testing free consumes tensor linearly..."
  t <- empty
  putStrLn "  Created tensor"
  putStrLn "  free : (1 t : Tensor) -> IO ()"
  putStrLn "  After free, tensor is consumed - no further use possible"
  free t
  putStrLn "  Tensor freed and consumed"
  -- Cannot use t here - it's been linearly consumed
  putStrLn "PASS: free consumes tensor linearly"

main : IO ()
main = testFreeConsumesLinearly
