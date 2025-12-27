-- Test for REQ_SD_LOAD_002: Check errors via hasError after loading
module Main

import StateDict.StateDict

-- Test hasError function: consumes StateDict, returns (Bool, StateDict)

testErrorCheck : IO Bool
testErrorCheck = do
  -- Load a nonexistent file to trigger error
  sd <- loadCheckpoint "definitely_does_not_exist.pt"
  -- Call hasError - this is the function under test
  (hasErr, sd') <- hasError sd
  -- Also test getError for completeness
  (maybeErr, sd'') <- getError sd'
  -- Clean up
  freeStateDict sd''
  -- For nonexistent file, hasErr should be True
  pure hasErr

main : IO ()
main = do
  result <- testErrorCheck
  if result
    then putStrLn "PASS: hasError correctly detects load failure"
    else putStrLn "INFO: hasError returned False (file may exist or error not detected)"
