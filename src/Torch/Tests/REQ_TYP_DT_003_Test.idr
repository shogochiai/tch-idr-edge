-- Test for REQ_TYP_DT_003: intToDType conversion for FFI results
module Main

import Torch.Torch

testIntToDType : IO ()
testIntToDType = do
  putStrLn "Testing intToDType conversion..."
  putStrLn $ "  intToDType 6 = " ++ show (intToDType 6) ++ " (expected Just TFloat32)"
  putStrLn $ "  intToDType 7 = " ++ show (intToDType 7) ++ " (expected Just TFloat64)"
  putStrLn $ "  intToDType 99 = " ++ show (intToDType 99) ++ " (expected Nothing)"
  let valid = case (intToDType 6, intToDType 7, intToDType 99) of
                (Just TFloat32, Just TFloat64, Nothing) => True
                _ => False
  if valid
    then putStrLn "PASS: intToDType returns correct values"
    else putStrLn "FAIL: intToDType conversion error"

main : IO ()
main = testIntToDType
