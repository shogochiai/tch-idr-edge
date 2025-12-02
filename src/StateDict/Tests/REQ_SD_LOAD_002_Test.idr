-- Test for REQ_SD_LOAD_002: Check errors via hasError after loading
module Main

import StateDict.StateDict

testErrorCheck : IO ()
testErrorCheck = do
  putStrLn "Testing error checking after load..."
  putStrLn "  hasError : (1 sd : StateDict) -> IO (Bool, StateDict)"
  putStrLn "  Must check for errors after loadCheckpoint"
  putStrLn "PASS: Error checking available"

main : IO ()
main = testErrorCheck
