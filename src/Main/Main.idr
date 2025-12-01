||| tch-idr test entry point
||| Demonstrates linear tensor usage
module Main.Main

import Torch.Torch
import FFI.FFI
import Tensor.Tensor

||| Example: Create tensor, inspect, and free
||| This demonstrates the linear usage pattern:
||| 1. Allocate -> 2. Use (chained) -> 3. Free
testLinearTensor : IO ()
testLinearTensor = do
  putStrLn "=== Linear Tensor Test ==="

  -- 1. Allocate
  t <- empty

  -- 2. Use (chained operations)
  (defined, t') <- isDefined t
  putStrLn $ "Tensor defined: " ++ show defined

  (d, t'') <- dim t'
  putStrLn $ "Tensor dim: " ++ show d

  -- 3. Free (mandatory!)
  free t''

  putStrLn "=== Test Complete ==="

main : IO ()
main = testLinearTensor
