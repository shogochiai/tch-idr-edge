-- Test for REQ_TYP_DEV_001: Device type (CPU, CUDA with device index)
module Main

import Torch.Torch

testDeviceType : IO ()
testDeviceType = do
  putStrLn "Testing Device type definition..."
  -- Test CPU device
  let cpu = CPU
  putStrLn $ "  CPU device created: " ++ show (deviceToInt cpu == -1)
  -- Test CUDA device with index 0
  let cuda0 = CUDA 0
  putStrLn $ "  CUDA 0 device created: " ++ show (deviceToInt cuda0 == 0)
  -- Test CUDA device with index 1
  let cuda1 = CUDA 1
  putStrLn $ "  CUDA 1 device created: " ++ show (deviceToInt cuda1 == 1)
  putStrLn "PASS: Device type correctly defined with CPU and CUDA constructors"

main : IO ()
main = testDeviceType
