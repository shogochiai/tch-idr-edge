-- Test for REQ_FFI_DE_001: at_free binding for tensor deallocation
module Main

import FFI.FFI

testFreeTensor : IO ()
testFreeTensor = do
  putStrLn "Testing at_free binding..."
  ptr <- newTensor
  putStrLn "  Created tensor"
  freeTensor ptr
  putStrLn "  Called freeTensor (at_free)"
  putStrLn "PASS: at_free binding works"

main : IO ()
main = testFreeTensor
