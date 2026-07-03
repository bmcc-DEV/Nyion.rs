# swamp-gpu/Makefile
# Compile CUDA kernels into shared library for Rust FFI

NVCC := nvcc
ARCH := -arch=sm_75  # GTX 1650 Mobile (Turing)
FLAGS := -O3 --use_fast_math -Xcompiler -fPIC -shared
INCLUDES := -I/usr/include
LIBS := -lcuda -lcudart
TARGET := libswamp_gpu.so
SRCDIR := kernels

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCDIR)/fused_attention.cu
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDES) -o $@ $^ $(LIBS)
	@echo "=== Built $@ ==="
	@file $@
	@ls -lh $@

# Compile to PTX for inspection
ptx: $(SRCDIR)/fused_attention.cu
	$(NVCC) $(ARCH) -O3 --use_fast_math -ptx -o fused_attention.ptx $^
	@echo "=== Generated PTX ==="

clean:
	rm -f $(TARGET) fused_attention.ptx
