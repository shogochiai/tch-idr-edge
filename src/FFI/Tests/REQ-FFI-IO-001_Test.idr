-- Test for REQ-FFI-IO-001: at_print binding for tensor printing
module Main

import FFI.FFI

testPrintTensor : IO ()
testPrintTensor = do
  putStrLn "Testing at_print binding..."
  ptr <- newTensor
  putStrLn "  Calling printTensor..."
  printTensor ptr
  freeTensor ptr
  putStrLn "PASS: at_print binding works"

main : IO ()
main = testPrintTensor
