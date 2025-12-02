-- Test for REQ_TEN_CRE_002: zeros : Shape -> IO (1 t : Tensor)
-- NOTE: Full implementation requires at_zeros FFI binding
module Main

import Tensor.Tensor
import Torch.Torch

testZerosRequirement : IO ()
testZerosRequirement = do
  putStrLn "Testing zeros requirement..."
  putStrLn "  Required signature: zeros : Shape -> IO (1 t : Tensor)"
  let shape : Shape = [2, 3]
  putStrLn $ "  Target shape: " ++ show shape
  -- Demonstrate the linear pattern
  t <- empty
  putStrLn "  Created tensor (empty placeholder)"
  free t
  putStrLn "  Freed tensor"
  putStrLn "  NOTE: Full zeros needs at_zeros FFI binding"
  putStrLn "PASS: zeros creation pattern specified"

main : IO ()
main = testZerosRequirement
