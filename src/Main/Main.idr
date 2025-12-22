||| tch-idr test entry point
||| Demonstrates linear tensor usage
module Main.Main

import Torch.Torch
import FFI.FFI
import Tensor.Tensor
import StateDict.StateDict

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

||| Test Tier 5: Data Bridge (fromListInt64, fromListDouble)
testDataBridge : IO ()
testDataBridge = do
  putStrLn "\n=== Tier 5: Data Bridge Test ==="

  -- fromListInt64
  a <- fromListInt64 [1, 2, 3, 4, 5]
  putStrLn "fromListInt64 [1, 2, 3, 4, 5]:"
  a' <- printT a
  (d, a'') <- dim a'
  putStrLn $ "dim = " ++ show d ++ " (expected 1)"
  (s, a''') <- size a'' 0
  putStrLn $ "size(0) = " ++ show s ++ " (expected 5)"
  free a'''

  -- fromListDouble
  b <- fromListDouble [1.5, 2.5, 3.5]
  putStrLn "fromListDouble [1.5, 2.5, 3.5]:"
  b' <- printT b
  free b'

  putStrLn "=== Data Bridge Complete ==="

||| Test Tier 5: Reduction Operations (mean, sum)
testReductions : IO ()
testReductions = do
  putStrLn "\n=== Tier 5: Reduction Operations Test ==="

  -- Create 2x3 matrix of ones
  a <- ones2d 2 3
  putStrLn "a = ones(2, 3):"
  a' <- printT a

  -- Duplicate for multiple tests
  (a1, a2) <- dup a'

  -- Mean along dim 0
  m0 <- mean a1 0
  putStrLn "mean(a, dim=0) (should be [1, 1, 1]):"
  m0' <- printT m0
  free m0'

  -- Sum along dim 1
  s1 <- sum a2 1
  putStrLn "sum(a, dim=1) (should be [3, 3]):"
  s1' <- printT s1
  free s1'

  putStrLn "=== Reduction Operations Complete ==="

||| Test Tier 5: Scalar Operations (divScalar, mulScalar)
testScalarOps : IO ()
testScalarOps = do
  putStrLn "\n=== Tier 5: Scalar Operations Test ==="

  -- Create tensor [2, 4, 6]
  a <- fromListDouble [2.0, 4.0, 6.0]
  putStrLn "a = [2.0, 4.0, 6.0]:"
  a' <- printT a

  -- Duplicate for multiple tests
  (a1, a2) <- dup a'

  -- Divide by 2
  b <- divScalar a1 2.0
  putStrLn "divScalar(a, 2.0) (should be [1, 2, 3]):"
  b' <- printT b
  free b'

  -- Multiply by 0.5
  c <- mulScalar a2 0.5
  putStrLn "mulScalar(a, 0.5) (should be [1, 2, 3]):"
  c' <- printT c
  free c'

  putStrLn "=== Scalar Operations Complete ==="

||| Test STPM integration scenario
testStpmScenario : IO ()
testStpmScenario = do
  putStrLn "\n=== STPM Integration Scenario Test ==="

  -- Simulate token indices from tokenizer
  let tokenIds = [101, 2023, 2003, 1037, 3231, 102] -- [CLS] this is a test [SEP]
  indices <- fromListInt64 tokenIds
  putStrLn "Token indices (simulated):"
  indices' <- printT indices

  -- Simulate embedding lookup result: 6 tokens x 4 hidden dim
  embeddings <- randn2d 6 4
  putStrLn "Embeddings shape 6x4:"
  embeddings' <- printT embeddings

  -- Sequence pooling via mean over dim 0
  pooled <- mean embeddings' 0
  putStrLn "Pooled (mean over sequence, should be 1x4 or just 4):"
  pooled' <- printT pooled

  -- Attention scaling: divide by sqrt(d) where d=4
  let sqrtD = 2.0  -- sqrt(4) = 2
  scaled <- divScalar pooled' sqrtD
  putStrLn "Scaled by 1/sqrt(4):"
  scaled' <- printT scaled

  -- Extract final score
  (score, scaled'') <- item scaled'
  putStrLn $ "Final score (first element): " ++ show score

  -- Cleanup
  free indices'
  free scaled''

  putStrLn "=== STPM Integration Complete ==="

||| Test Tier 6: StateDict Loading (Checkpoint Support)
covering
testCheckpointLoading : IO ()
testCheckpointLoading = do
  putStrLn "\n=== Tier 6: Checkpoint Loading Test ==="

  -- Load the STPM checkpoint (absolute path for reliability)
  let checkpointPath = "/Users/bob/code/eventhorizon-and-lazysolidity/eventhorizon/crates/stpm/models/checkpoint_epoch_11500_dataset_10k_context_512_cpu_state_dict.pt"
  putStrLn $ "Loading checkpoint: " ++ checkpointPath

  sd <- loadCheckpoint checkpointPath

  -- Check for errors
  (hasErr, sd') <- hasError sd
  case hasErr of
    True => do putStrLn "ERROR: Failed to load checkpoint (check stderr for details)"
               freeStateDict sd'
    False => do
      -- Get tensor count
      (n, sd'') <- sdSize sd'
      putStrLn $ "Loaded " ++ show n ++ " tensors"

      -- Get embedding weight tensor as example
      putStrLn "Extracting 'model_state_dict.embed.weight'..."
      (mEmb, sd''') <- getTensor sd'' "model_state_dict.embed.weight"
      case mEmb of
        Nothing => do
          putStrLn "  Tensor not found"
          freeStateDict sd'''
        Just emb => do
          (d, emb') <- dim emb
          putStrLn $ "  dim = " ++ show d
          (s0, emb'') <- Tensor.size emb' 0
          putStrLn $ "  size(0) = " ++ show s0 ++ " (expected 259 = vocab size)"
          (s1, emb''') <- Tensor.size emb'' 1
          putStrLn $ "  size(1) = " ++ show s1 ++ " (expected 512 = hidden dim)"
          free emb'''
          freeStateDict sd'''

      putStrLn "=== Checkpoint Loading Complete ==="

||| Test dupDeep (deep clone)
testDupDeep : IO ()
testDupDeep = do
  putStrLn "\n=== DupDeep (Deep Clone) Test ==="

  -- Create a tensor
  a <- ones1d 3
  putStrLn "a = ones(3):"
  a' <- printT a

  -- Deep clone it
  putStrLn "Deep cloning a..."
  (b, c) <- dupDeep a'
  putStrLn "b (first clone):"
  b' <- printT b
  putStrLn "c (second clone):"
  c' <- printT c

  -- Free b first
  putStrLn "Freeing b..."
  free b'

  -- c should still be valid
  putStrLn "c after freeing b (should still work):"
  c'' <- printT c'
  (d, c''') <- dim c''
  putStrLn $ "c.dim = " ++ show d

  -- Free c
  free c'''

  putStrLn "=== DupDeep Test Complete ==="

||| Test cloneBorrow (borrow semantics)
testCloneBorrow : IO ()
testCloneBorrow = do
  putStrLn "\n=== CloneBorrow Test ==="

  -- Create a tensor
  a <- ones1d 3
  putStrLn "a = ones(3):"
  a' <- printT a

  -- Clone using borrow semantics (doesn't consume a')
  putStrLn "Cloning a (borrow semantics)..."
  b <- cloneBorrow a'
  putStrLn "b (clone):"
  b' <- printT b

  -- a' should still be valid
  putStrLn "a after clone (should still work):"
  a'' <- printT a'
  (d, a''') <- dim a''
  putStrLn $ "a.dim = " ++ show d

  -- Clone again to test multiple borrows
  putStrLn "Cloning a again..."
  c <- cloneBorrow a'''
  putStrLn "c (second clone):"
  c' <- printT c

  -- Free all
  free a'''
  free b'
  free c'

  putStrLn "=== CloneBorrow Test Complete ==="

main : IO ()
main = do
  -- Basic tests
  testLinearTensor
  testDebug
  testArithmetic
  testMatmul
  testShapeOps
  testCreation
  testQueries
  testCombination
  testNNPrimitives
  testDataBridge
  testReductions
  testScalarOps
  testStpmScenario
  -- Test deep clone
  testDupDeep
  -- Test borrow clone
  testCloneBorrow
  -- Tier 6: Checkpoint loading
  testCheckpointLoading
  putStrLn "All tests complete!"
