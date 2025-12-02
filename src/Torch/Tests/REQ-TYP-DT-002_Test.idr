-- Test for REQ-TYP-DT-002: dtypeToInt conversion for FFI calls
module Main

import Torch.Torch

testDTypeToInt : IO ()
testDTypeToInt = do
  putStrLn "Testing dtypeToInt conversion..."
  putStrLn $ "  dtypeToInt TFloat32 = " ++ show (dtypeToInt TFloat32) ++ " (expected 6)"
  putStrLn $ "  dtypeToInt TFloat64 = " ++ show (dtypeToInt TFloat64) ++ " (expected 7)"
  putStrLn $ "  dtypeToInt TInt32 = " ++ show (dtypeToInt TInt32) ++ " (expected 3)"
  putStrLn $ "  dtypeToInt TInt64 = " ++ show (dtypeToInt TInt64) ++ " (expected 4)"
  putStrLn $ "  dtypeToInt TBool = " ++ show (dtypeToInt TBool) ++ " (expected 11)"
  let allCorrect = dtypeToInt TFloat32 == 6 &&
                   dtypeToInt TFloat64 == 7 &&
                   dtypeToInt TInt32 == 3 &&
                   dtypeToInt TInt64 == 4 &&
                   dtypeToInt TBool == 11
  if allCorrect
    then putStrLn "PASS: dtypeToInt returns correct values"
    else putStrLn "FAIL: dtypeToInt conversion error"

main : IO ()
main = testDTypeToInt
