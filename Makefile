NVCC ?= nvcc
CUDA_PATH ?= /usr/local/cuda
CXXFLAGS := -std=c++17 -O2 -I$(CUDA_PATH)/include
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -lnppif -lnppc

TARGET := batch_processor
SRC := src/batch_processor.cu

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p results
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET) results/*.txt results/*.csv results/*.pgm

.PHONY: all clean