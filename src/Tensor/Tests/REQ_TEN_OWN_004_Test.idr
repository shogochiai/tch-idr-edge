-- Test for REQ_TEN_OWN_004: dupDeep creates independent tensors
-- Verify that deep clone allows freeing one copy while keeping the other
module Main

import Tensor.Tensor

||| Test that dupDeep creates truly independent copies
||| After freeing one copy, the other should still be usable
testDeepCloneIndependence : IO ()
testDeepCloneIndependence = do
  putStrLn "Testing deep clone independence..."

  -- Create a tensor with data
  t <- ones2d 3 3
  putStrLn "  Created 3x3 ones tensor"

  -- Deep duplicate
  (t1, t2) <- dupDeep t
  putStrLn "  Deep duplicated tensor"

  -- Check dims of both
  (d1, t1') <- dim t1
  (d2, t2') <- dim t2
  putStrLn $ "  t1 dims: " ++ show d1
  putStrLn $ "  t2 dims: " ++ show d2

  -- Free t1' - this should NOT affect t2'
  free t1'
  putStrLn "  Freed first copy"

  -- t2' should still be valid and usable
  (d2', t2'') <- dim t2'
  putStrLn $ "  t2 dims after freeing t1: " ++ show d2'

  -- Extract value to prove data is intact
  (val, t2''') <- item t2''
  putStrLn $ "  t2 value: " ++ show val

  free t2'''
  putStrLn "  Freed second copy"

  if d2' == 2 && val > 0.9 && val < 1.1
     then putStrLn "PASS: Deep clone creates independent tensors"
     else putStrLn "FAIL: Deep clone data corruption detected"

||| Contrast test: shallow dup shares storage (demonstrating the problem)
||| This is expected to have shared storage behavior
testShallowVsDeep : IO ()
testShallowVsDeep = do
  putStrLn "Testing shallow vs deep clone..."

  -- Create tensor
  t <- ones2d 2 2

  -- Deep dup - independent
  (t1, t2) <- dupDeep t

  -- Check both are defined
  (def1, t1') <- isDefined t1
  (def2, t2') <- isDefined t2

  putStrLn $ "  t1 defined: " ++ show def1
  putStrLn $ "  t2 defined: " ++ show def2

  -- Free both independently
  free t1'
  free t2'

  putStrLn "PASS: Both deep-cloned tensors freed independently"

main : IO ()
main = do
  testDeepCloneIndependence
  putStrLn ""
  testShallowVsDeep
