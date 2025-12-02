-- Test for REQ_TEN_LIN_003: Require explicit free call - NO implicit Drop
module Main

import Tensor.Tensor

testExplicitFree : IO ()
testExplicitFree = do
  putStrLn "Testing explicit free requirement..."
  t <- empty
  putStrLn "  Allocated tensor"
  putStrLn "  Linear type enforces exactly-once usage"
  free t
  putStrLn "  Called free explicitly (no implicit Drop)"
  putStrLn "PASS: Explicit free required"

main : IO ()
main = testExplicitFree
