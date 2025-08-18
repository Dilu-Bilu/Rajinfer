MAKEFLAGS+=-r -j

UNAME=$(shell uname)

NVCC?=nvcc

BUILD=build

# compile .c, .cpp, .cu files
SOURCES=$(wildcard src/*.c)
SOURCES+=$(wildcard src/*.cc)
SOURCES+=$(wildcard src/*.cpp)
SOURCES+=$(wildcard src/*.cu)
SOURCES+=$(wildcard vendor/*.c)
SOURCES+=$(wildcard vendor/*.cc)
SOURCES+=$(wildcard vendor/*.cpp)
SOURCES+=$(wildcard vendor/*.cu)

OBJECTS=$(SOURCES:%=$(BUILD)/%.o)
BINARY=$(BUILD)/main

CFLAGS=-g -Wall -Wpointer-arith -Werror -O3 -ffast-math -Ivendor -std=c++20
LDFLAGS=-lm

CFLAGS+=-fopenmp -mf16c -mavx2 -mfma
LDFLAGS+=-fopenmp
LDFLAGS+=-lcudart

ifneq (,$(wildcard /usr/local/cuda))
  LDFLAGS+=-L/usr/local/cuda/lib64
endif

CUFLAGS+=-g -O2 -lineinfo
CUFLAGS+=-allow-unsupported-compiler # for recent CUDA versions

ifeq ($(CUARCH),)
  CUFLAGS+=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 --threads 2
else
  CUFLAGS+=-arch=$(CUARCH)
endif

all: $(BINARY)

format:
	clang-format -i src/*

$(BINARY): $(OBJECTS)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/%.c.o: %.c
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cc.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $< $(CFLAGS) -c -MMD -MP -o $@

$(BUILD)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $< $(CUFLAGS) -c -MMD -MP -o $@

-include $(OBJECTS:.o=.d)

clean:
	rm -rf $(BUILD)

.PHONY: all clean format