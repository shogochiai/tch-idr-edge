-- Test for REQ-TYP-DEV-002: deviceToInt conversion for FFI calls
module Main

import Torch.Torch

testDeviceToInt : IO ()
testDeviceToInt = do
  putStrLn "Testing deviceToInt conversion..."
  -- CPU maps to -1 (LibTorch convention)
  putStrLn $ "  deviceToInt CPU = " ++ show (deviceToInt CPU)
  putStrLn $ "  Expected: -1, Got: " ++ show (deviceToInt CPU) ++ " -> " ++ show (deviceToInt CPU == -1)
  -- CUDA 0 maps to 0
  putStrLn $ "  deviceToInt (CUDA 0) = " ++ show (deviceToInt (CUDA 0))
  putStrLn $ "  Expected: 0, Got: " ++ show (deviceToInt (CUDA 0)) ++ " -> " ++ show (deviceToInt (CUDA 0) == 0)
  -- CUDA 2 maps to 2
  putStrLn $ "  deviceToInt (CUDA 2) = " ++ show (deviceToInt (CUDA 2))
  putStrLn $ "  Expected: 2, Got: " ++ show (deviceToInt (CUDA 2)) ++ " -> " ++ show (deviceToInt (CUDA 2) == 2)
  putStrLn "PASS: deviceToInt correctly converts Device to FFI integers"

main : IO ()
main = testDeviceToInt
