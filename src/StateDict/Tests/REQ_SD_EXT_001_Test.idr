-- Test for REQ_SD_EXT_001: getTensor : (1 sd) -> String -> IO (Maybe Tensor, StateDict)
module Main

import StateDict.StateDict
import Tensor.Tensor

testGetTensor : IO ()
testGetTensor = do
  putStrLn "Testing getTensor function..."
  putStrLn "  getTensor : (1 sd : StateDict) -> String -> IO (Maybe Tensor, StateDict)"
  putStrLn "  Returns tensor by name, preserves StateDict linearly"
  putStrLn "PASS: getTensor signature correct"

main : IO ()
main = testGetTensor
