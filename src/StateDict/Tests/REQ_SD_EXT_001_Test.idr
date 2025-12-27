-- Test for REQ_SD_EXT_001: getTensor : (1 sd) -> String -> IO (Maybe Tensor, StateDict)
module Main

import StateDict.StateDict
import Tensor.Tensor

-- Test getTensor function call and return type

testGetTensor : IO Bool
testGetTensor = do
  -- Load a StateDict (will have error but that's OK for this test)
  sd <- loadCheckpoint "nonexistent.pt"
  -- Call getTensor - this is the function under test
  (maybeTensor, sd') <- getTensor sd "some_tensor_name"
  -- Handle the Maybe result
  case maybeTensor of
    Nothing => do
      freeStateDict sd'
      pure True  -- Expected for nonexistent file
    Just t => do
      free t
      freeStateDict sd'
      pure True

main : IO ()
main = do
  result <- testGetTensor
  if result
    then putStrLn "PASS: getTensor is callable and returns (Maybe Tensor, StateDict)"
    else putStrLn "FAIL: getTensor test failed"
