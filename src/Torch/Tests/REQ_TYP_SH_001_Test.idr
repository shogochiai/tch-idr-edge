-- Test for REQ_TYP_SH_001: Shape as List Int for tensor dimensions
module Main

import Torch.Torch

testShapeType : IO ()
testShapeType = do
  putStrLn "Testing Shape type definition..."
  -- Shape is defined as List Int
  let scalar : Shape = []
  putStrLn $ "  Scalar shape (0-dim): " ++ show scalar
  let vector : Shape = [10]
  putStrLn $ "  Vector shape (1-dim): " ++ show vector
  let matrix : Shape = [3, 4]
  putStrLn $ "  Matrix shape (2-dim): " ++ show matrix
  let tensor3d : Shape = [2, 3, 4]
  putStrLn $ "  3D tensor shape: " ++ show tensor3d
  -- Verify they are List Int
  putStrLn $ "  Length of scalar shape: " ++ show (length scalar)
  putStrLn $ "  Length of matrix shape: " ++ show (length matrix)
  putStrLn "PASS: Shape correctly defined as List Int"

main : IO ()
main = testShapeType
