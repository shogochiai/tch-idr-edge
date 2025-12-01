||| tch-idr test entry point
||| Demonstrates linear tensor usage
module Main.Main

import Torch.Torch
import FFI.FFI
import Tensor.Tensor

||| Example: Create tensor, inspect, and free
||| This demonstrates the linear usage pattern:
||| 1. Allocate -> 2. Use (chained) -> 3. Free
testLinearTensor : IO ()
testLinearTensor = do
  putStrLn "=== Linear Tensor Test ==="

  -- 1. Allocate
  t <- empty

  -- 2. Use (chained operations)
  (defined, t') <- isDefined t
  putStrLn $ "Tensor defined: " ++ show defined

  (d, t'') <- dim t'
  putStrLn $ "Tensor dim: " ++ show d

  -- 3. Free (mandatory!)
  free t''

  putStrLn "=== Test Complete ==="

||| Test arithmetic operations with linear types
testArithmetic : IO ()
testArithmetic = do
  putStrLn "\n=== Arithmetic Test ==="

  -- Create two 1D tensors
  a <- ones1d 4
  b <- ones1d 4

  -- Print inputs
  putStrLn "a = ones(4):"
  a' <- printT a
  putStrLn "b = ones(4):"
  b' <- printT b

  -- Add them (consumes a' and b', produces c)
  c <- add a' b'
  putStrLn "c = a + b:"
  c' <- printT c

  -- Get dimension
  (d, c'') <- dim c'
  putStrLn $ "c.dim = " ++ show d

  -- Free result
  free c''

  putStrLn "=== Arithmetic Complete ==="

||| Test matrix multiplication
testMatmul : IO ()
testMatmul = do
  putStrLn "\n=== Matmul Test ==="

  -- Create 2x3 and 3x2 matrices
  a <- ones2d 2 3
  b <- ones2d 3 2

  putStrLn "a = ones(2, 3):"
  a' <- printT a
  putStrLn "b = ones(3, 2):"
  b' <- printT b

  -- Matmul: (2x3) @ (3x2) = (2x2)
  c <- matmul a' b'
  putStrLn "c = a @ b (should be 2x2 of 3s):"
  c' <- printT c

  free c'

  putStrLn "=== Matmul Complete ==="

||| Debug FFI test
testDebug : IO ()
testDebug = do
  putStrLn "\n=== Debug FFI Test ==="

  -- Test simple echo
  putStrLn "Testing debugEcho(42)..."
  result <- debugEcho 42
  putStrLn $ "Result: " ++ show result ++ " (expected 84)"

  -- Test out-parameter pattern
  putStrLn "Testing debugOutptr(10)..."
  ptr <- debugOutptr 10
  putStrLn $ "Result ptr: (should be 30)"

  putStrLn "=== Debug Complete ==="

||| Test Tier 1: Shape operations
testShapeOps : IO ()
testShapeOps = do
  putStrLn "\n=== Tier 1: Shape Operations Test ==="

  -- Create 2x3 matrix
  a <- ones2d 2 3
  putStrLn "a = ones(2, 3):"
  a' <- printT a

  -- Transpose to 3x2
  b <- transpose a' 0 1
  putStrLn "b = transpose(a, 0, 1) (should be 3x2):"
  b' <- printT b

  -- Reshape to 6
  c <- reshape2d b' 1 6
  putStrLn "c = reshape(b, 1, 6):"
  c' <- printT c

  free c'
  putStrLn "=== Shape Operations Complete ==="

||| Test Tier 2: Tensor creation
testCreation : IO ()
testCreation = do
  putStrLn "\n=== Tier 2: Tensor Creation Test ==="

  -- Random tensor
  a <- randn2d 2 3
  putStrLn "a = randn(2, 3):"
  a' <- printT a

  free a'
  putStrLn "=== Creation Complete ==="

||| Test Tier 3: Shape queries
testQueries : IO ()
testQueries = do
  putStrLn "\n=== Tier 3: Shape Queries Test ==="

  -- Create 3x4 matrix
  a <- ones2d 3 4
  putStrLn "a = ones(3, 4)"

  -- Query dimensions
  (s0, a') <- size a 0
  putStrLn $ "size(a, 0) = " ++ show s0 ++ " (expected 3)"

  (s1, a'') <- size a' 1
  putStrLn $ "size(a, 1) = " ++ show s1 ++ " (expected 4)"

  (d, a''') <- dim a''
  putStrLn $ "dim(a) = " ++ show d ++ " (expected 2)"

  free a'''
  putStrLn "=== Queries Complete ==="

||| Test Tier 4: Tensor combination
testCombination : IO ()
testCombination = do
  putStrLn "\n=== Tier 4: Tensor Combination Test ==="

  -- Create two 2x3 matrices
  a <- ones2d 2 3
  b <- ones2d 2 3

  -- Cat along dim 0 -> 4x3
  c <- cat2 a b 0
  putStrLn "cat([ones(2,3), ones(2,3)], dim=0) (should be 4x3):"
  c' <- printT c

  free c'

  -- Stack creates new dimension
  d <- ones1d 3
  e <- ones1d 3
  f <- stack2 d e 0
  putStrLn "stack([ones(3), ones(3)], dim=0) (should be 2x3):"
  f' <- printT f

  free f'
  putStrLn "=== Combination Complete ==="

||| Test Tier 5: Neural network primitives
testNNPrimitives : IO ()
testNNPrimitives = do
  putStrLn "\n=== Tier 5: NN Primitives Test ==="

  -- Layer norm
  a <- randn2d 2 4
  putStrLn "a = randn(2, 4):"
  a' <- printT a

  b <- layerNorm a' 4
  putStrLn "layerNorm(a, 4) (normalized over last dim):"
  b' <- printT b
  free b'

  -- GELU
  c <- randn1d 4
  putStrLn "c = randn(4):"
  c' <- printT c

  d <- gelu c'
  putStrLn "gelu(c):"
  d' <- printT d
  free d'

  -- Dropout (training=False should be identity)
  e <- ones1d 4
  putStrLn "e = ones(4):"
  e' <- printT e

  f <- dropout e' 0.5 False
  putStrLn "dropout(e, 0.5, training=False) (should be same as input):"
  f' <- printT f
  free f'

  putStrLn "=== NN Primitives Complete ==="

main : IO ()
main = do
  testLinearTensor
  testDebug
  testArithmetic
  testMatmul
  testShapeOps
  testCreation
  testQueries
  testCombination
  testNNPrimitives
