-- Test for REQ-FFI-CR-001: at_new_tensor binding for empty tensor creation
module Main

import FFI.FFI

testNewTensor : IO ()
testNewTensor = do
  putStrLn "Testing at_new_tensor binding..."
  putStrLn "  Calling newTensor..."
  ptr <- newTensor
  putStrLn "  Tensor pointer obtained"
  putStrLn "  Freeing tensor..."
  freeTensor ptr
  putStrLn "  Tensor freed"
  putStrLn "PASS: at_new_tensor binding works"

main : IO ()
main = testNewTensor
