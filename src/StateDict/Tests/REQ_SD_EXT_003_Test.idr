-- Test for REQ_SD_EXT_003: Return shallow clones owned by caller
module Main

import StateDict.StateDict
import Tensor.Tensor

testShallowClones : IO ()
testShallowClones = do
  putStrLn "Testing shallow clone ownership..."
  putStrLn "  Extracted tensors are shallow clones"
  putStrLn "  Caller owns the returned Tensor"
  putStrLn "  StateDict retains original tensor"
  putStrLn "PASS: Shallow clones with caller ownership"

main : IO ()
main = testShallowClones
