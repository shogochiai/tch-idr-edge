# tch-idr-edge by lazy-idris

**Explicit Linear Type Bindings for LibTorch (CPU-only, Inference-only)**

> ⚠️ **Edge Deployment Focus**: This library is designed for CPU-only inference on edge devices. No GPU/CUDA support, no training capabilities.

## STI Parity Status

| Metric | Score |
|--------|-------|
| ST Parity (Spec-Test) | 100% (37/37) |
| SI Parity (Spec-Impl) | 100% |
| T-I Parity (Test-Impl) | 100% (37/37) |

Run audit: `DYLD_LIBRARY_PATH=".:$(brew --prefix libtorch)/lib" lazy-idris audit .`

This project ports a **minimal subset** of [tch-rs](https://github.com/LaurentMazare/tch-rs) to Idris 2, replacing Rust's affine types (implicit `Drop`) with Idris 2's **Linear Types**. This ensures that all resource management—especially tensor deallocation—is explicit, verifiable, and free from "invisible" compiler actions that plague autonomous software evolution.

## Limitations (vs tch-rs)

**tch-idr-edge is NOT a full port of tch-rs.** It is a minimal, CPU-only inference library implementing only ~40 operations needed for lazy-idris STPM inference:

### Design Constraints
- **CPU-only**: No GPU/CUDA support. Optimized for edge deployment where GPU is unavailable.
- **Inference-only**: No autograd, no training. Forward pass only.
- **Limited tensor operations**: Only operations required for STPM inference are implemented.

| Feature | tch-rs | tch-idr-edge |
|---------|--------|---------|
| Tensor operations | ~2000+ | ~40 |
| Autograd | ✅ | ❌ |
| Model loading (.pt) | ✅ | ✅ (StateDict only) |
| GPU/CUDA support | ✅ | ❌ |
| Optimizers | ✅ | ❌ |
| Full NN modules | ✅ | ❌ (minimal primitives only) |

**Supported operations (41 total):**

| Category | Operations |
|----------|------------|
| **Creation** | `empty`, `zeros1d`, `zeros2d`, `ones1d`, `ones2d`, `randn1d`, `randn2d`, `randn3d`, `fromListInt64`, `fromListDouble` |
| **Arithmetic** | `add`, `mul`, `matmul`, `divScalar`, `mulScalar` |
| **Shape** | `transpose`, `reshape2d`, `reshape3d`, `reshape4d`, `view2d`, `view3d`, `view4d`, `cat2`, `stack2` |
| **Slicing** | `narrow`, `split3` |
| **NN primitives** | `embedding`, `layerNorm`, `softmax`, `relu`, `gelu`, `dropout` |
| **Reduction** | `mean`, `sum` |
| **Query** | `dim`, `size`, `item`, `isDefined`, `printT` |
| **Memory** | `dup`, `free` |
| **StateDict** | `loadStateDict`, `stateDictLen`, `stateDictKeyByIndex`, `stateDictTensorByName`, `freeStateDict` |

If you need full LibTorch functionality, use tch-rs directly.

## 1\. Architectural Overview

We strictly separate the system into three layers to maintain the **Chain of Trust**:

1.  **Layer 0: C++ Shim (`torch-sys`)**
      * **Source:** Direct extraction from `tch-rs/torch-sys/libtch`.
      * **Role:** Exposes LibTorch (C++) classes as C-compatible functions.
      * **Artifact:** `libtorch_shim.so` (or `.dylib`).
2.  **Layer 1: Raw FFI (`Torch.FFI`)**
      * **Role:** Unsafe `PrimIO` bindings to the C++ Shim.
      * **Type:** `Ptr Any -> PrimIO ()`. No safety guarantees here.
3.  **Layer 2: Linear Wrapper (`Torch.Tensor`)**
      * **Role:** The **lazy-idris** safety layer. Wraps raw pointers in linear types.
      * **Invariant:** `(1 t : Tensor)` ensures exactly-once usage. Explicit `free` is mandatory.

## 2\. Prerequisites & Environment

  * **Idris 2** (v0.6.0+)
  * **LibTorch** (C++ Library, corresponding to the version in `tch-rs`)
  * **Clang++** (Required for compiling the shim)
  * **Make**

### Environment Variables

The development environment must have:

```bash
export LIBTORCH_PATH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$LIBTORCH_PATH/include:$LIBTORCH_PATH/include/torch/csrc/api/include
```

## 3\. Directory Structure Specification

The LLM Agent must enforce this directory structure:

```text
.
├── Makefile                # Orchestrates C++ and Idris builds
├── idris-torch.ipkg        # Idris package definition
├── torch-sys/              # C++ Source (Shim)
│   ├── torch_api.h         # Copied from tch-rs
│   ├── torch_api.cpp       # Copied from tch-rs
│   └── wrappers/           # Additional C wrappers if needed
└── src/
    ├── Main.idr            # Entry point / Integration Test
    └── Torch/
        ├── FFI.idr         # Layer 1: Raw Bindings
        ├── Tensor.idr      # Layer 2: Linear Types
        └── Types.idr       # Common Types (Scalar, DType)
```

## 4\. Implementation Procedures

This project follows **Strategy B (Cowboy-style)** defined in the *lazy-idris* whitepaper, but guided by strict type constraints.

### Phase 1: The Shim (Extraction)

**Goal:** Create `libtorch_shim.so`.

1.  Locate `torch_api.h` and `torch_api.cpp` in the `tch-rs` source tree.
2.  Copy them to `./torch-sys/`.
3.  Create a `Makefile` target `shim` to compile these into a shared object.
      * *Constraint:* Must link against `torch`, `torch_cpu`, `c10`.

### Phase 2: FFI Bootstrap (Layer 1)

**Goal:** Verify connection to LibTorch.

1.  In `src/Torch/FFI.idr`, define the foreign import for tensor creation and deletion.
      * Target: `at_tensor_of_data`, `at_print`, `at_free`.
2.  Create `src/Main.idr` to call these functions using `unsafePerformIO` (temporary) just to verify the link.

### Phase 3: Linearization (Layer 2)

**Goal:** Implement the "No Implicit Drop" policy.

1.  Define the opaque type in `src/Torch/Tensor.idr`:
    ```idris
    export
    data Tensor : Type where
      MkTensor : AnyPtr -> Tensor
    ```
2.  Implement the **Linear Interface**:
      * **Creation:** `makeTensor : List Double -> IO (1 t : Tensor)`
      * **Usage:** `print : (1 t : Tensor) -> IO (1 t : Tensor)`
      * **Destruction:** `free : (1 t : Tensor) -> IO ()`

**Crucial Logic for LLM:**

  * If Rust code has `fn foo(&self)`, Idris must be `foo : (1 t : Tensor) -> IO (1 t : Tensor)`. (Borrowing becomes chaining).
  * If Rust code has `Drop`, Idris must have `free`.
  * If Rust code returns `Tensor`, Idris returns `(1 _ : Tensor)`.

### Phase 4: Expansion (Iterative)

Follow the `tch-rs` module structure to expand functionality.

1.  **Ops:** Implement `add`, `mul`, `matmul`.
      * Rust: `t1 + t2` (Implicitly creates new tensor, keeps old ones).
      * Idris: `add : (1 t1 : Tensor) -> (1 t2 : Tensor) -> IO (Res (1 t1 : Tensor) (1 t2 : Tensor) (1 out : Tensor))`
      * *Note:* Since arithmetic operations in Torch usually copy, we need to return the original linear resources *plus* the new one, OR explicitly consume them. **Decision:** Follow the C++ API semantics explicitly.
2.  **NN:** Implement `Linear` layer.
      * Structs in Rust (`struct Linear { ws: Tensor, bs: Tensor }`) must become Linear Records in Idris holding Linear Tensors.

## 5\. Development Workflow for Agents

Agents must follow this loop:

1.  **Read Rust Source:** Open `tch-rs/src/tensor/ops.rs` (or target file).
2.  **Identify C-Shim:** Find the corresponding function in `torch_api.h`.
3.  **Bind Raw:** Add `%foreign` entry to `FFI.idr`.
4.  **Bind Linear:** Add linear wrapper to `Tensor.idr`.
5.  **Test:** Add a test case in `src/Main.idr` that:
      * Allocates a tensor.
      * Performs the operation.
      * **Frees all tensors.**
6.  **Verify:** Run `make test`. If it compiles, resource safety is mathematically guaranteed.

## 6\. Build Instructions

To build the shim and the Idris project:

```bash
# 1. Compile C++ Shim
make shim

# 2. Build Idris Interface
make build

# 3. Run Tests
make test
```

### Makefile Template

```makefile
LIBTORCH_PATH ?= /usr/local/libtorch
CXX ?= clang++
IDRIS ?= idris2

CFLAGS = -fPIC -std=c++14 \
         -I$(LIBTORCH_PATH)/include \
         -I$(LIBTORCH_PATH)/include/torch/csrc/api/include

LDFLAGS = -L$(LIBTORCH_PATH)/lib \
          -ltorch -ltorch_cpu -lc10 \
          -Wl,-rpath,$(LIBTORCH_PATH)/lib

TARGET_SHIM = libtorch_shim.so

shim:
	$(CXX) $(CFLAGS) -shared -o $(TARGET_SHIM) torch-sys/torch_api.cpp $(LDFLAGS)

build:
	$(IDRIS) --build idris-torch.ipkg

test: shim
	$(IDRIS) --build idris-torch.ipkg
	./build/exec/run_tests
```

-----

**Directive to LLM:**
Start by executing **Phase 1**. Do not proceed to Phase 2 until `libtorch_shim.so` is successfully generated.
