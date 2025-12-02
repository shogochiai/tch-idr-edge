||| Linear Type StateDict Wrapper
||| Layer 2: Safe checkpoint loading for lazy-idris
||| Invariant: (1 sd : StateDict) ensures exactly-once usage
module StateDict.StateDict

import FFI.FFI
import Tensor.Tensor

%default total

||| Opaque state dict type with linear constraint
||| Represents a loaded PyTorch checkpoint
export
data StateDict : Type where
  MkStateDict : StateDictPtr -> StateDict

-- ============================================================
-- StateDict Loading
-- ============================================================

||| Load state dict from a PyTorch checkpoint file (.pt)
||| Returns: Linear StateDict that MUST be freed
|||
||| Note: Check for errors with hasError after loading
||| The checkpoint should be a pure tensor archive (e.g., from prepare_weights.py)
export
loadCheckpoint : String -> IO StateDict
loadCheckpoint path = do
  ptr <- loadStateDictRaw path
  pure (MkStateDict ptr)

-- ============================================================
-- StateDict Destruction (Linear Consuming)
-- Defined early so it can be used by other functions
-- ============================================================

||| Free state dict and all contained tensors
||| Linear: Consumes the state dict, no further use allowed
||| Note: Any tensors extracted via getTensor/tensorAt are NOT freed
|||       (they are shallow clones that the caller owns)
export
freeStateDict : (1 sd : StateDict) -> IO ()
freeStateDict (MkStateDict ptr) = freeStateDictRaw ptr

-- ============================================================
-- StateDict Inspection (Linear Preserving)
-- ============================================================

||| Check if state dict has an error
||| Linear: Returns result AND the state dict
export
hasError : (1 sd : StateDict) -> IO (Bool, StateDict)
hasError (MkStateDict ptr) = do
  merr <- stateDictError ptr
  case merr of
    Nothing => pure (False, MkStateDict ptr)
    Just _  => pure (True, MkStateDict ptr)

||| Get error message if any
||| Linear: Returns result AND the state dict
export
getError : (1 sd : StateDict) -> IO (Maybe String, StateDict)
getError (MkStateDict ptr) = do
  merr <- stateDictError ptr
  pure (merr, MkStateDict ptr)

||| Get number of tensors in state dict
||| Linear: Returns result AND the state dict
export
sdSize : (1 sd : StateDict) -> IO (Nat, StateDict)
sdSize (MkStateDict ptr) = do
  n <- stateDictSize ptr
  pure (cast n, MkStateDict ptr)

||| Check if state dict is empty (no tensors or load failed)
||| Linear: Returns result AND the state dict
export
isEmpty : (1 sd : StateDict) -> IO (Bool, StateDict)
isEmpty (MkStateDict ptr) = do
  n <- stateDictSize ptr
  pure (n == 0, MkStateDict ptr)

||| Get tensor name at index
||| Linear: Returns result AND the state dict
||| Returns Nothing if index out of bounds
export
nameAt : (1 sd : StateDict) -> Nat -> IO (Maybe String, StateDict)
nameAt (MkStateDict ptr) idx = do
  mname <- stateDictNameAt ptr (cast idx)
  pure (mname, MkStateDict ptr)

-- ============================================================
-- Tensor Extraction (Linear Preserving)
-- ============================================================

||| Get tensor by name from state dict
||| Returns a shallow clone - caller owns and must free the returned tensor
||| Linear: Returns (Maybe tensor, state dict)
export
getTensor : (1 sd : StateDict) -> String -> IO (Maybe Tensor, StateDict)
getTensor (MkStateDict ptr) name = do
  tPtr <- stateDictTensorByName ptr name
  if prim__nullAnyPtr tPtr /= 0
     then pure (Nothing, MkStateDict ptr)
     else pure (Just (MkTensor tPtr), MkStateDict ptr)

||| Get tensor at index from state dict
||| Returns a shallow clone - caller owns and must free the returned tensor
||| Linear: Returns (Maybe tensor, state dict)
export
tensorAt : (1 sd : StateDict) -> Nat -> IO (Maybe Tensor, StateDict)
tensorAt (MkStateDict ptr) idx = do
  tPtr <- stateDictTensorAt ptr (cast idx)
  if prim__nullAnyPtr tPtr /= 0
     then pure (Nothing, MkStateDict ptr)
     else pure (Just (MkTensor tPtr), MkStateDict ptr)

-- ============================================================
-- Batch Extraction (Linear Consuming)
-- ============================================================

||| Extract all tensors as a list with names
||| Linear: Consumes state dict, returns list of (name, tensor) pairs
||| All returned tensors MUST be freed individually by the caller
export covering
extractAll : (1 sd : StateDict) -> IO (List (String, Tensor))
extractAll (MkStateDict ptr) = do
  n <- stateDictSize ptr
  extractLoop (MkStateDict ptr) 0 (cast n) []
  where
    covering
    extractLoop : (1 sd : StateDict) -> Nat -> Nat -> List (String, Tensor) -> IO (List (String, Tensor))
    extractLoop (MkStateDict p) idx max acc =
      case idx >= max of
        True => do freeStateDictRaw p
                   pure (reverse acc)
        False => do mname <- stateDictNameAt p (cast idx)
                    tPtr <- stateDictTensorAt p (cast idx)
                    case (mname, prim__nullAnyPtr tPtr /= 0) of
                      (Just name, False) => extractLoop (MkStateDict p) (idx + 1) max ((name, MkTensor tPtr) :: acc)
                      _ => extractLoop (MkStateDict p) (idx + 1) max acc
