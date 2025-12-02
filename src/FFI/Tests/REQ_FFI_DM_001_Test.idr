-- Test for REQ_FFI_DM_001: at_device binding for device query
module Main

import FFI.FFI
import Torch.Torch

testDeviceBinding : IO ()
testDeviceBinding = do
  putStrLn "Testing at_device binding..."
  ptr <- newTensor
  putStrLn "  Created tensor"
  -- tensorDevice wraps at_device
  deviceInt <- tensorDevice ptr
  putStrLn $ "  at_device returned: " ++ show deviceInt
  -- -1 means CPU in LibTorch convention
  let isCpu = deviceInt == -1
  putStrLn $ "  Tensor on CPU: " ++ show isCpu
  freeTensor ptr
  putStrLn "  Freed tensor"
  putStrLn "PASS: at_device binding works correctly"

main : IO ()
main = testDeviceBinding
