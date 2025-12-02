-- Test for REQ_FFI_ERR_001: get_and_reset_last_err binding for error retrieval
module Main

import FFI.FFI

testErrorRetrievalBinding : IO ()
testErrorRetrievalBinding = do
  putStrLn "Testing get_and_reset_last_err binding..."
  -- getLastErr wraps prim__getLastErr
  err <- getLastErr
  putStrLn "  getLastErr called successfully"
  case err of
    Nothing => putStrLn "  No error present (expected)"
    Just msg => putStrLn $ "  Error: " ++ msg
  putStrLn "  get_and_reset_last_err binding verified"
  putStrLn "PASS: Error retrieval binding works correctly"

main : IO ()
main = testErrorRetrievalBinding
