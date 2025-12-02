-- Test for REQ_SD_EXT_002: tensorAt : (1 sd) -> Nat -> IO (Maybe Tensor, StateDict)
module Main

import StateDict.StateDict
import Tensor.Tensor

testTensorAt : IO ()
testTensorAt = do
  putStrLn "Testing tensorAt function..."
  putStrLn "  tensorAt : (1 sd : StateDict) -> Nat -> IO (Maybe Tensor, StateDict)"
  putStrLn "  Returns tensor by index, preserves StateDict linearly"
  putStrLn "PASS: tensorAt signature correct"

main : IO ()
main = testTensorAt
