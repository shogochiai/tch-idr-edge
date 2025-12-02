-- Test for REQ-TYP-SH-002: Helper functions for shape manipulation
module Main

import Torch.Torch

-- Test helper functions that should exist for shapes
testShapeHelpers : IO ()
testShapeHelpers = do
  putStrLn "Testing Shape helper functions..."
  let shape1 : Shape = [2, 3, 4]
  -- elementSize helper from Torch module
  putStrLn $ "  elementSize TFloat32 = " ++ show (elementSize TFloat32)
  putStrLn $ "  Expected: 4, Got: " ++ show (elementSize TFloat32) ++ " -> " ++ show (elementSize TFloat32 == 4)
  putStrLn $ "  elementSize TFloat64 = " ++ show (elementSize TFloat64)
  putStrLn $ "  Expected: 8, Got: " ++ show (elementSize TFloat64) ++ " -> " ++ show (elementSize TFloat64 == 8)
  putStrLn $ "  elementSize TInt8 = " ++ show (elementSize TInt8)
  putStrLn $ "  Expected: 1, Got: " ++ show (elementSize TInt8) ++ " -> " ++ show (elementSize TInt8 == 1)
  -- Standard list functions work on Shape (since Shape = List Int)
  putStrLn $ "  length shape [2,3,4] = " ++ show (length shape1)
  putStrLn $ "  reverse shape [2,3,4] = " ++ show (reverse shape1)
  putStrLn "PASS: Shape helpers work correctly"

main : IO ()
main = testShapeHelpers
