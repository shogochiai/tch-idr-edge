-- Test for REQ_TEN_INS_003: print : (1 t : Tensor) -> IO (1 t : Tensor)
module Main

import Tensor.Tensor

testPrintInspection : IO ()
testPrintInspection = do
  putStrLn "Testing print inspection..."
  putStrLn "  printT : (1 t : Tensor) -> IO Tensor"
  t <- empty
  putStrLn "  Created tensor"
  putStrLn "  Calling printT:"
  t' <- printT t
  putStrLn "  printT returned tensor for continued use"
  free t'
  putStrLn "  Tensor freed"
  putStrLn "PASS: print preserves tensor linearly"

main : IO ()
main = testPrintInspection
