-- Test for REQ-MAIN-001: Demonstrate linear tensor usage pattern
module Main

import Tensor.Tensor

testLinearUsagePattern : IO ()
testLinearUsagePattern = do
  putStrLn "Testing linear tensor usage pattern..."
  putStrLn "  Pattern: create -> use -> free"
  -- Demonstrate the canonical pattern
  t <- empty
  putStrLn "  1. Created tensor"
  (b, t') <- isDefined t
  putStrLn $ "  2. Used tensor (isDefined: " ++ show b ++ ")"
  free t'
  putStrLn "  3. Freed tensor"
  putStrLn "  Linear pattern: exactly-once usage enforced"
  putStrLn "PASS: Linear tensor usage pattern demonstrated"

main : IO ()
main = testLinearUsagePattern
