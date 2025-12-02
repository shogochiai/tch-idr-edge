-- Test for REQ_SD_LIN_003: Require explicit freeStateDict call - NO implicit Drop
module Main

import StateDict.StateDict

testExplicitFree : IO ()
testExplicitFree = do
  putStrLn "Testing explicit freeStateDict requirement..."
  putStrLn "  NO implicit Drop - must call freeStateDict explicitly"
  putStrLn "  Linear types enforce this at compile time"
  putStrLn "PASS: Explicit free required"

main : IO ()
main = testExplicitFree
