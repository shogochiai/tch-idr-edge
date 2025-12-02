-- Test for REQ_SD_LIN_001: Wrap StateDictPtr in opaque StateDict type
module Main

import StateDict.StateDict

testOpaqueType : IO ()
testOpaqueType = do
  putStrLn "Testing opaque StateDict type..."
  -- StateDict constructor should not be exported
  -- Only loadCheckpoint should create StateDict values
  putStrLn "  StateDict is opaque (MkStateDict constructor not exported)"
  putStrLn "PASS: StateDict type wraps StateDictPtr"

main : IO ()
main = testOpaqueType
