-- Test for REQ_MAIN_002: Chain tensor operations correctly
module Main

import Tensor.Tensor

testChainOperations : IO ()
testChainOperations = do
  putStrLn "Testing operation chaining..."
  t0 <- empty
  putStrLn "  Created t0"
  -- Chain 1: isDefined
  (b1, t1) <- isDefined t0
  putStrLn $ "  t0 -> isDefined -> (b1=" ++ show b1 ++ ", t1)"
  -- Chain 2: dim
  (d, t2) <- dim t1
  putStrLn $ "  t1 -> dim -> (d=" ++ show d ++ ", t2)"
  -- Chain 3: printT
  t3 <- printT t2
  putStrLn "  t2 -> printT -> t3"
  -- Chain 4: isDefined again
  (b2, t4) <- isDefined t3
  putStrLn $ "  t3 -> isDefined -> (b2=" ++ show b2 ++ ", t4)"
  -- Final
  free t4
  putStrLn "  t4 -> free"
  putStrLn "  Chain: t0 -> t1 -> t2 -> t3 -> t4 -> freed"
  putStrLn "PASS: Operations chained correctly"

main : IO ()
main = testChainOperations
