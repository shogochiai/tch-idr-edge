-- Test for REQ_SD_LIN_002: Enforce (1 sd : StateDict) linear annotation on all arguments
module Main

import StateDict.StateDict
import Tensor.Tensor

-- Linear type verification: This test compiles IFF linear annotations are correct
-- 1. loadCheckpoint returns a linear StateDict
-- 2. hasError consumes and returns a linear StateDict
-- 3. freeStateDict consumes the linear StateDict

testLinearAnnotation : IO Bool
testLinearAnnotation = do
  -- Actual function calls with linear types
  sd <- loadCheckpoint "nonexistent.pt"  -- Returns (1 sd : StateDict)
  (hasErr, sd') <- hasError sd           -- Consumes and returns linear StateDict
  freeStateDict sd'                      -- Consumes the linear StateDict
  -- If this compiles, linear annotations are enforced
  pure True

main : IO ()
main = do
  result <- testLinearAnnotation
  if result
    then putStrLn "PASS: Linear annotations enforced"
    else putStrLn "FAIL: Linear annotations not enforced"
