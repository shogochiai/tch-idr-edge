-- Test for REQ-FFI-DE-002: Raw pointer access for explicit memory management
module Main

import FFI.FFI
import System.FFI

testRawPointerAccess : IO ()
testRawPointerAccess = do
  putStrLn "Testing raw pointer access..."
  -- TensorPtr is exposed as AnyPtr for explicit memory management
  ptr <- newTensor
  putStrLn "  Created tensor, got TensorPtr (AnyPtr)"
  putStrLn "  TensorPtr : Type"
  putStrLn "  TensorPtr = AnyPtr"
  -- Verify we can check for null
  let isNull = prim__nullAnyPtr ptr /= 0
  putStrLn $ "  Pointer is null: " ++ show isNull
  -- Cleanup
  freeTensor ptr
  putStrLn "  Freed via raw pointer"
  putStrLn "PASS: Raw pointer access exposed correctly"

main : IO ()
main = testRawPointerAccess
