LIBTORCH_PATH ?= /usr/local/libtorch
CXX ?= clang++
IDRIS ?= idris2

CFLAGS = -fPIC -std=c++17 \
         -I$(LIBTORCH_PATH)/include \
         -I$(LIBTORCH_PATH)/include/torch/csrc/api/include \
         -Itorch-sys \
         -D_GLIBCXX_USE_CXX11_ABI=0

LDFLAGS = -L$(LIBTORCH_PATH)/lib \
          -ltorch -ltorch_cpu -lc10 \
          -Wl,-rpath,$(LIBTORCH_PATH)/lib

# Use .dylib on macOS, .so on Linux
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    TARGET_SHIM = libtorch_shim.dylib
else
    TARGET_SHIM = libtorch_shim.so
endif

.PHONY: all shim build test clean

all: shim build

shim: $(TARGET_SHIM)

$(TARGET_SHIM): torch-sys/torch_api.cpp torch-sys/torch_api_generated.cpp torch-sys/stubs.cpp
	$(CXX) $(CFLAGS) -shared -o $@ $^ $(LDFLAGS)

build: shim
	$(IDRIS) --build tch-idr-edge.ipkg

OMP_PATH ?= /opt/homebrew/Cellar/llvm/20.1.7/lib

test: build
ifeq ($(UNAME_S),Darwin)
	DYLD_LIBRARY_PATH=$(LIBTORCH_PATH)/lib:$(OMP_PATH):. ./build/exec/tch-idr-edge-test
else
	LD_LIBRARY_PATH=$(LIBTORCH_PATH)/lib:. ./build/exec/tch-idr-edge-test
endif

clean:
	rm -rf build/
	rm -f $(TARGET_SHIM)

# Development helpers
check:
	$(IDRIS) --check tch-idr-edge.ipkg

repl:
	$(IDRIS) --repl tch-idr-edge.ipkg

# Copy shim sources from cargo cache (one-time setup)
extract-shim:
	@echo "Copying torch-sys sources..."
	cp ~/.cargo/registry/src/index.crates.io-*/torch-sys-*/libtch/torch_api.h torch-sys/
	cp ~/.cargo/registry/src/index.crates.io-*/torch-sys-*/libtch/torch_api.cpp torch-sys/
	cp ~/.cargo/registry/src/index.crates.io-*/torch-sys-*/libtch/torch_api_generated.h torch-sys/
	cp ~/.cargo/registry/src/index.crates.io-*/torch-sys-*/libtch/torch_api_generated.cpp torch-sys/
