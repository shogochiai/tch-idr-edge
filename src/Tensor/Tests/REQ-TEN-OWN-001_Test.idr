-- Test for REQ-TEN-OWN-001: Return (1 _ : Tensor) for operations preserving tensor
module Main

import Tensor.Tensor

testLinearChaining : IO ()
testLinearChaining = do
  putStrLn "Testing linear chaining..."
  t <- empty
  putStrLn "  Created tensor"
  (b, t') <- isDefined t
  putStrLn $ "  isDefined returned: " ++ show b
  putStrLn "  Tensor preserved through operation"
  (d, t'') <- dim t'
  putStrLn $ "  dim returned: " ++ show d
  free t''
  putStrLn "  Freed final tensor"
  putStrLn "PASS: Operations preserve tensor linearly"

main : IO ()
main = testLinearChaining
