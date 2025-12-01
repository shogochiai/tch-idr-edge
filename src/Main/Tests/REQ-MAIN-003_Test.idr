-- Test for REQ-MAIN-003: Free all allocated tensors
module Main

import Tensor.Tensor

testFreeAllTensors : IO ()
testFreeAllTensors = do
  putStrLn "Testing free all allocated tensors..."
  -- Allocate multiple tensors
  t1 <- empty
  putStrLn "  Allocated t1"
  t2 <- empty
  putStrLn "  Allocated t2"
  t3 <- empty
  putStrLn "  Allocated t3"
  -- Free all of them
  free t1
  putStrLn "  Freed t1"
  free t2
  putStrLn "  Freed t2"
  free t3
  putStrLn "  Freed t3"
  putStrLn "  All 3 tensors allocated and freed"
  putStrLn "PASS: All allocated tensors freed"

main : IO ()
main = testFreeAllTensors
