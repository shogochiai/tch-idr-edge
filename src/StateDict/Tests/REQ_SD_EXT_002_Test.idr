-- Test for REQ_SD_EXT_002: tensorAt : (1 sd) -> Nat -> IO (Maybe Tensor, StateDict)
module Main

import StateDict.StateDict
import Tensor.Tensor

-- Test tensorAt function call and return type

testTensorAt : IO Bool
testTensorAt = do
  -- Load a StateDict
  sd <- loadCheckpoint "nonexistent.pt"
  -- Call tensorAt - this is the function under test
  (maybeTensor, sd') <- tensorAt sd 0
  -- Handle the Maybe result
  case maybeTensor of
    Nothing => do
      freeStateDict sd'
      pure True  -- Expected for empty/error StateDict
    Just t => do
      free t
      freeStateDict sd'
      pure True

main : IO ()
main = do
  result <- testTensorAt
  if result
    then putStrLn "PASS: tensorAt is callable and returns (Maybe Tensor, StateDict)"
    else putStrLn "FAIL: tensorAt test failed"
