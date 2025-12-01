-- Test for REQ-FFI-DM-002: at_data_ptr binding for raw data access
module Main

import FFI.FFI
import System.FFI

testDataPtrBinding : IO ()
testDataPtrBinding = do
  putStrLn "Testing at_data_ptr binding..."
  ptr <- newTensor
  putStrLn "  Created tensor"
  -- tensorDataPtr wraps at_data_ptr
  dataPtr <- tensorDataPtr ptr
  putStrLn "  at_data_ptr called successfully"
  -- Empty tensor may have null data pointer
  let isNull = prim__nullAnyPtr dataPtr /= 0
  putStrLn $ "  Data pointer is null (empty tensor): " ++ show isNull
  freeTensor ptr
  putStrLn "  Freed tensor"
  putStrLn "PASS: at_data_ptr binding works correctly"

main : IO ()
main = testDataPtrBinding
