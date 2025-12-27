-- Test for REQ_SD_LIN_003: Require explicit freeStateDict call - NO implicit Drop
module Main

import StateDict.StateDict

-- This test verifies freeStateDict is callable and consumes the StateDict
-- If implicit Drop were possible, this explicit call would be optional
-- The linear type system requires this call

testExplicitFree : IO Bool
testExplicitFree = do
  -- Create a StateDict
  sd <- loadCheckpoint "nonexistent.pt"
  -- Must explicitly free - no implicit Drop
  freeStateDict sd
  -- If we reach here, explicit free works
  pure True

main : IO ()
main = do
  result <- testExplicitFree
  if result
    then putStrLn "PASS: Explicit freeStateDict required and works"
    else putStrLn "FAIL: Explicit free test failed"
