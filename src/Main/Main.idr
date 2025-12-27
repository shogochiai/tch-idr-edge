||| tch-idr test entry point
||| Demonstrates linear tensor usage
module Main.Main

import Torch.Torch
import FFI.FFI
import Tensor.Tensor
import StateDict.StateDict
import System.File
import Data.String
import Data.List
import Data.List1

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

-- ============================================================
-- File-based tensor loading for isolation test
-- ============================================================

-- Memory allocation
%foreign "C:malloc,libc 6"
prim__mallocMain : Bits64 -> PrimIO AnyPtr

%foreign "C:free,libc 6"
prim__freePtrMain : AnyPtr -> PrimIO ()

-- File reading
%foreign "C:fopen,libc 6"
prim__fopen : String -> String -> PrimIO AnyPtr

%foreign "C:fclose,libc 6"
prim__fclose : AnyPtr -> PrimIO Int

%foreign "C:fread,libc 6"
prim__fread : AnyPtr -> Bits64 -> Bits64 -> AnyPtr -> PrimIO Bits64

%foreign "C:fseek,libc 6"
prim__fseek : AnyPtr -> Int64 -> Int -> PrimIO Int

%foreign "C:ftell,libc 6"
prim__ftell : AnyPtr -> PrimIO Int64

-- Write int64 to buffer
%foreign "C:idris_write_int64,libtorch_shim"
prim__writeInt64 : AnyPtr -> Bits64 -> Bits64 -> PrimIO ()

-- Tensor creation from data
%foreign "C:at_tensor_of_data,libtorch_shim"
prim__tensorOfDataMain : AnyPtr -> AnyPtr -> Bits64 -> Bits64 -> Int -> PrimIO AnyPtr

-- Deep clone
%foreign "C:at_deep_clone,libtorch_shim"
prim__deepCloneMain : AnyPtr -> PrimIO AnyPtr

||| Get file size
getFileSizeMain : AnyPtr -> IO Bits64
getFileSizeMain fp = do
  _ <- primIO (prim__fseek fp 0 2)  -- SEEK_END = 2
  size <- primIO (prim__ftell fp)
  _ <- primIO (prim__fseek fp 0 0)  -- SEEK_SET = 0
  pure (cast size)

||| Load raw binary tensor from file (minimal implementation for testing)
||| Returns Nothing on failure
covering
loadTestTensor : String -> String -> IO (Maybe Tensor)
loadTestTensor binPath shapePath = do
  -- Read shape file (hardcoded parsing for simplicity)
  Right shapeContent <- readFile shapePath
    | Left _ => pure Nothing

  -- Parse shape - assume format "d0,d1" for 2D
  let trimmed = trim shapeContent
  let parts = split (== ',') trimmed
  let dims : List Bits64 = mapMaybe (\s => map cast (parseInteger {a=Integer} (trim s))) (forget parts)

  if null dims
     then pure Nothing
     else do
       -- Open binary file
       fp <- primIO (prim__fopen binPath "rb")
       if prim__nullAnyPtr fp /= 0
          then pure Nothing
          else do
            -- Get file size and allocate buffer
            fileSize <- getFileSizeMain fp
            dataBuf <- primIO (prim__mallocMain fileSize)
            if prim__nullAnyPtr dataBuf /= 0
               then do
                 _ <- primIO (prim__fclose fp)
                 pure Nothing
               else do
                 -- Read data
                 _ <- primIO (prim__fread dataBuf 1 fileSize fp)
                 _ <- primIO (prim__fclose fp)

                 -- Create dims array
                 let ndims = cast {to=Bits64} (length dims)
                 dimsBuf <- primIO (prim__mallocMain (ndims * 8))  -- 8 bytes per int64
                 writeDimsMain dimsBuf 0 dims

                 -- Create tensor (dtype 6 = Float32, element_size = 4)
                 tmpPtr <- primIO (prim__tensorOfDataMain dataBuf dimsBuf ndims 4 6)

                 -- Free dims buffer - libtorch copies this
                 primIO (prim__freePtrMain dimsBuf)

                 if prim__nullAnyPtr tmpPtr /= 0
                    then do
                      primIO (prim__freePtrMain dataBuf)
                      pure Nothing
                    else do
                      -- Deep clone to make tensor own its storage
                      clonedPtr <- primIO (prim__deepCloneMain tmpPtr)

                      -- Free the temporary tensor
                      freeTensor tmpPtr
                      -- Free the data buffer
                      primIO (prim__freePtrMain dataBuf)
                      pure (Just (MkTensor clonedPtr))
  where
    writeDimsMain : AnyPtr -> Bits64 -> List Bits64 -> IO ()
    writeDimsMain _ _ [] = pure ()
    writeDimsMain buf idx (d :: ds) = do
      primIO (prim__writeInt64 buf idx d)
      writeDimsMain buf (idx + 1) ds

||| Test file-loaded tensors with matmul (isolation test for TRM bug)
covering
testFileLoadMatmul : IO ()
testFileLoadMatmul = do
  putStrLn "\n=== File Load + Matmul Test ==="

  let testDir = "/Users/bob/code/tch-idr-edge/test_tensors"

  -- Test 1: Load small tensors (512-dim, known to work)
  putStrLn "Test 1: Load & matmul 513x512 (small)..."
  mInput513 <- loadTestTensor (testDir ++ "/input_513.bin") (testDir ++ "/input_513.shape")
  mWeight513 <- loadTestTensor (testDir ++ "/weight_513x512.bin") (testDir ++ "/weight_513x512.shape")
  mBias512 <- loadTestTensor (testDir ++ "/bias_512.bin") (testDir ++ "/bias_512.shape")

  case (mInput513, mWeight513, mBias512) of
    (Just input, Just weight, Just bias) => do
      putStrLn "  Loaded all tensors"
      (s0, input') <- size input 0
      (s1, input'') <- size input' 1
      putStrLn $ "  input shape: [" ++ show s0 ++ ", " ++ show s1 ++ "]"

      projected <- matmul input'' weight
      putStrLn "  matmul done"
      result <- add projected bias
      putStrLn "  add done"
      free result
      putStrLn "  Test 1 PASS"
    _ => putStrLn "  SKIP: Failed to load tensors"

  -- Test 2: Load large tensors (1024-dim, the failing case)
  putStrLn "Test 2: Load & matmul 1025x1024 (large - the TRM failing case)..."
  mInput1025 <- loadTestTensor (testDir ++ "/input_1025.bin") (testDir ++ "/input_1025.shape")
  mWeight1024 <- loadTestTensor (testDir ++ "/weight_1025x1024.bin") (testDir ++ "/weight_1025x1024.shape")
  mBias1024 <- loadTestTensor (testDir ++ "/bias_1024.bin") (testDir ++ "/bias_1024.shape")

  case (mInput1025, mWeight1024, mBias1024) of
    (Just input, Just weight, Just bias) => do
      putStrLn "  Loaded all tensors"
      (s0, input') <- size input 0
      (s1, input'') <- size input' 1
      putStrLn $ "  input shape: [" ++ show s0 ++ ", " ++ show s1 ++ "]"

      putStrLn "  Calling matmul..."
      projected <- matmul input'' weight
      putStrLn "  matmul returned, checking result..."

      -- This is where TRM crashes - let's see if it happens here too
      (pDim, projected') <- dim projected
      putStrLn $ "  projected dim: " ++ show pDim

      (ps0, projected'') <- size projected' 0
      (ps1, projected''') <- size projected'' 1
      putStrLn $ "  projected shape: [" ++ show ps0 ++ ", " ++ show ps1 ++ "]"

      putStrLn "  Calling add..."
      result <- add projected''' bias
      putStrLn "  add done"
      free result
      putStrLn "  Test 2 PASS"
    _ => putStrLn "  SKIP: Failed to load tensors"

  -- Test 3: cat2 with file-loaded tensors then matmul
  putStrLn "Test 3: File load -> cat2 -> matmul..."
  mA3 <- loadTestTensor (testDir ++ "/bias_512.bin") (testDir ++ "/bias_512.shape")
  mB3 <- loadTestTensor (testDir ++ "/bias_512.bin") (testDir ++ "/bias_512.shape")
  mY3 <- pure (Just ()) >>= (\_ => fromListDouble [0.5] >>= (\t => view2d t 1 1 >>= (\t' => pure (Just t'))))
  mWt3 <- loadTestTensor (testDir ++ "/weight_1025x1024.bin") (testDir ++ "/weight_1025x1024.shape")
  mBias3 <- loadTestTensor (testDir ++ "/bias_1024.bin") (testDir ++ "/bias_1024.shape")

  case (mA3, mB3, mY3, mWt3, mBias3) of
    (Just a, Just b, Just y, Just wt, Just bias) => do
      -- Deep clone all loaded tensors before cat2 to ensure they're fully independent
      putStrLn "  Deep cloning input tensors..."
      aClone <- cloneBorrow a
      bClone <- cloneBorrow b
      free a
      free b

      putStrLn "  cat2(aClone, bClone)..."
      ab <- cat2 aClone bClone 1
      putStrLn "  cat2(ab, y)..."
      aby <- cat2 ab y 1
      (s0, aby') <- size aby 0
      (s1, aby'') <- size aby' 1
      putStrLn $ "  aby shape: [" ++ show s0 ++ ", " ++ show s1 ++ "]"

      -- Deep clone the concatenated result
      putStrLn "  Deep cloning aby..."
      abyClone <- cloneBorrow aby''
      free aby''

      putStrLn "  Calling matmul..."
      projected <- matmul abyClone wt
      putStrLn "  matmul returned, checking result..."

      (pDim, projected') <- dim projected
      putStrLn $ "  projected dim: " ++ show pDim

      (ps0, projected'') <- size projected' 0
      (ps1, projected''') <- size projected'' 1
      putStrLn $ "  projected shape: [" ++ show ps0 ++ ", " ++ show ps1 ++ "]"

      putStrLn "  Calling add..."
      result <- add projected''' bias
      putStrLn "  add done"
      free result
      putStrLn "  Test 3 PASS"
    _ => putStrLn "  SKIP: Failed to load tensors (Test 3)"

  putStrLn "=== File Load + Matmul Test Complete ==="

||| Test large tensor operations (1024 dimensions)
||| This tests the TRM inference failure scenario
testLargeTensor : IO ()
testLargeTensor = do
  putStrLn "\n=== Large Tensor Test (1024 dim) ==="

  -- Test 1: Simple 512-dim add (known to work)
  putStrLn "Test 1: add [1,512] + [1,512]..."
  a512 <- ones2d 1 512
  b512 <- ones2d 1 512
  c512 <- add a512 b512
  putStrLn "  add 512-dim: OK"
  free c512

  -- Test 2: Simple 1024-dim add
  putStrLn "Test 2: add [1,1024] + [1,1024]..."
  a1024 <- ones2d 1 1024
  b1024 <- ones2d 1 1024
  c1024 <- add a1024 b1024
  putStrLn "  add 1024-dim: OK"
  free c1024

  -- Test 3: cat2 to create 1024-dim, then add
  putStrLn "Test 3: cat2 [1,512]+[1,512] -> [1,1024], then add..."
  x1 <- ones2d 1 512
  x2 <- ones2d 1 512
  catted <- cat2 x1 x2 1  -- [1, 1024]
  bias1 <- ones2d 1 1024
  result1 <- add catted bias1
  putStrLn "  cat2 + add: OK"
  free result1

  -- Test 4: matmul [1,1025] @ [1025,1024] -> [1,1024]
  putStrLn "Test 4: matmul [1,1025] @ [1025,1024]..."
  input <- ones2d 1 1025
  weight <- ones2d 1025 1024
  projected <- matmul input weight
  putStrLn "  matmul: OK"

  -- Test 5: add after matmul (the failing case in TRM)
  putStrLn "Test 5: add [1,1024] + [1,1024] after matmul..."
  bias2 <- ones2d 1 1024
  result2 <- add projected bias2
  putStrLn "  add after matmul: OK"
  free result2

  -- Test 6: Full pipeline: cat2 -> matmul -> add (TRM Think Linear0)
  putStrLn "Test 6: Full pipeline cat2 -> matmul -> add..."
  p1 <- ones2d 1 512
  p2 <- ones2d 1 512
  p3 <- ones2d 1 1
  temp1 <- cat2 p1 p2 1     -- [1, 1024]
  input2 <- cat2 temp1 p3 1 -- [1, 1025]
  w2 <- ones2d 1025 1024
  proj2 <- matmul input2 w2 -- [1, 1024]
  b2 <- ones2d 1 1024
  final <- add proj2 b2     -- [1, 1024]
  putStrLn "  Full pipeline: OK"
  free final

  putStrLn "=== Large Tensor Test Complete ==="

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
  -- Large tensor test (1024 dim)
  testLargeTensor
  -- File load + matmul test (TRM bug isolation)
  testFileLoadMatmul
  putStrLn "All tests complete!"
