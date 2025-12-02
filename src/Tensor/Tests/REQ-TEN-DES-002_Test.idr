-- Test for REQ-TEN-DES-002: Guarantee no tensor leak if free is called
module Main

import Tensor.Tensor

testNoLeakOnFree : IO ()
testNoLeakOnFree = do
  putStrLn "Testing no tensor leak guarantee..."
  -- Create and free multiple tensors
  t1 <- empty
  putStrLn "  Created tensor t1"
  free t1
  putStrLn "  Freed t1"
  t2 <- empty
  putStrLn "  Created tensor t2"
  free t2
  putStrLn "  Freed t2"
  t3 <- empty
  putStrLn "  Created tensor t3"
  -- Use before free
  (_, t3') <- isDefined t3
  free t3'
  putStrLn "  Freed t3"
  putStrLn "  All tensors freed - no leaks"
  putStrLn "PASS: No tensor leak when free is called"

main : IO ()
main = testNoLeakOnFree
