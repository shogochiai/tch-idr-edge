-- Test for REQ_TEN_OWN_003: Chain operations returning both original and result tensors
module Main

import Tensor.Tensor

testChainedOperations : IO ()
testChainedOperations = do
  putStrLn "Testing chained operations with tensor threading..."
  t0 <- empty
  putStrLn "  Created tensor t0"
  -- Chain isDefined: returns (Bool, Tensor)
  (b1, t1) <- isDefined t0
  putStrLn $ "  isDefined t0 -> (b1=" ++ show b1 ++ ", t1)"
  -- Chain dim: returns (Nat, Tensor)
  (d, t2) <- dim t1
  putStrLn $ "  dim t1 -> (d=" ++ show d ++ ", t2)"
  -- Chain printT: returns Tensor
  t3 <- printT t2
  putStrLn "  printT t2 -> t3"
  -- Chain isDefined again
  (b2, t4) <- isDefined t3
  putStrLn $ "  isDefined t3 -> (b2=" ++ show b2 ++ ", t4)"
  -- Final free
  free t4
  putStrLn "  free t4 - chain complete"
  putStrLn "PASS: Operations correctly chain tensor ownership"

main : IO ()
main = testChainedOperations
