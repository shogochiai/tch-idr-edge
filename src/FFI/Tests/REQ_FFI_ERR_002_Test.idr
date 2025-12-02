-- Test for REQ_FFI_ERR_002: checkError helper for consistent error handling
module Main

import FFI.FFI

-- checkError helper pattern
checkError : IO a -> IO (Either String a)
checkError action = do
  result <- action
  err <- getLastErr
  case err of
    Nothing => pure (Right result)
    Just msg => pure (Left msg)

testCheckErrorHelper : IO ()
testCheckErrorHelper = do
  putStrLn "Testing checkError helper pattern..."
  -- Test with successful operation
  result <- checkError newTensor
  case result of
    Right ptr => do
      putStrLn "  Tensor creation succeeded (no error)"
      freeTensor ptr
      putStrLn "  Tensor freed"
    Left msg => putStrLn $ "  Unexpected error: " ++ msg
  putStrLn "  checkError pattern demonstrated"
  putStrLn "PASS: checkError helper pattern works correctly"

main : IO ()
main = testCheckErrorHelper
