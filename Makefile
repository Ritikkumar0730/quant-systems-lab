# Quant Systems Lab build: plain GNU make with any C11/C++20 compiler (clang or gcc).
#
#   make              build libraries, test binaries and benchmarks (-O3)
#   make test         run the C, C++, interview and Python test suites
#   make sanitize     rebuild under AddressSanitizer + UBSan and run native tests
#   make tsan         rebuild under ThreadSanitizer and run the C++ tests
#   make bench        run the order-book and SPSC latency/throughput benchmarks
#   make clean

ifeq ($(origin CC),default)
  CC := cc
endif
ifeq ($(origin CXX),default)
  CXX := c++
endif

BUILD  ?= build
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

ifeq ($(shell uname -s),Darwin)
  SHARED_EXT     := dylib
  SHARED_LDFLAGS := -dynamiclib
else
  SHARED_EXT     := so
  SHARED_LDFLAGS := -shared
endif

OPT      ?= -O3 -DNDEBUG
SANITIZE ?=

WARN     := -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror
CFLAGS   := -std=c11 $(WARN) -Wstrict-prototypes -Wmissing-prototypes $(OPT) $(SANITIZE) \
            -fPIC -Ic/include -MMD -MP
CXXFLAGS := -std=c++20 $(WARN) -Wsign-conversion -Wold-style-cast -Wnon-virtual-dtor \
            -Woverloaded-virtual $(OPT) $(SANITIZE) -fPIC -Icpp/include -Ic/include -MMD -MP
LDFLAGS  := $(SANITIZE)
LDLIBS   := -lm
THREADS  := -pthread

OBJ := $(BUILD)/obj
BIN := $(BUILD)/bin
LIB := $(BUILD)/lib

C_OBJS         := $(patsubst %.c,$(OBJ)/%.o,$(wildcard c/src/*.c))
CPP_API_OBJS   := $(patsubst %.cpp,$(OBJ)/%.o,$(wildcard cpp/src/*.cpp))
CPP_TEST_OBJS  := $(patsubst %.cpp,$(OBJ)/%.o,$(wildcard cpp/tests/*.cpp))
INTV_CPP_OBJS  := $(patsubst %.cpp,$(OBJ)/%.o,$(wildcard interview/cpp/*.cpp))
INTV_C_OBJS    := $(patsubst %.c,$(OBJ)/%.o,$(wildcard interview/c/*.c))

LIB_C_STATIC   := $(LIB)/libqsl.a
LIB_C_SHARED   := $(LIB)/libqsl.$(SHARED_EXT)
LIB_CPP_SHARED := $(LIB)/libqslcpp.$(SHARED_EXT)

.PHONY: all c cpp interview test test-native test-c test-cpp test-interview test-py \
        sanitize tsan bench clean help

all: c cpp interview

c: $(LIB_C_SHARED) $(BIN)/test_c
cpp: $(LIB_CPP_SHARED) $(BIN)/test_cpp $(BIN)/bench_lob $(BIN)/bench_spsc
interview: $(BIN)/test_interview_cpp $(BIN)/test_interview_c

# ------------------------------------------------------------------ objects
$(OBJ)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---------------------------------------------------------------- libraries
$(LIB_C_STATIC): $(C_OBJS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

$(LIB_C_SHARED): $(C_OBJS)
	@mkdir -p $(@D)
	$(CC) $(SHARED_LDFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# C ABI over the header-only C++ engines (order book, trees, PDE, Heston) for Python.
$(LIB_CPP_SHARED): $(CPP_API_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(SHARED_LDFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS) $(THREADS)

# ----------------------------------------------------------------- binaries
$(BIN)/test_c: $(OBJ)/c/tests/test_qsl.o $(LIB_C_STATIC)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BIN)/test_cpp: $(CPP_TEST_OBJS) $(LIB_C_STATIC)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS) $(THREADS)

$(BIN)/bench_lob: $(OBJ)/cpp/bench/bench_order_book.o
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS) $(THREADS)

$(BIN)/bench_spsc: $(OBJ)/cpp/bench/bench_spsc.o
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS) $(THREADS)

$(BIN)/test_interview_cpp: $(INTV_CPP_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BIN)/test_interview_c: $(INTV_C_OBJS)
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# -------------------------------------------------------------------- tests
test: test-native test-py

test-native: test-c test-cpp test-interview

test-c: $(BIN)/test_c
	$(BIN)/test_c

test-cpp: $(BIN)/test_cpp
	$(BIN)/test_cpp

test-interview: $(BIN)/test_interview_cpp $(BIN)/test_interview_c
	$(BIN)/test_interview_cpp
	$(BIN)/test_interview_c

test-py: $(LIB_C_SHARED) $(LIB_CPP_SHARED)
	$(PYTHON) -m pytest

sanitize:
	$(MAKE) BUILD=build/asan OPT="-O1 -g -fno-omit-frame-pointer" \
	    SANITIZE="-fsanitize=address,undefined -fno-sanitize-recover=all" test-native

tsan:
	$(MAKE) BUILD=build/tsan OPT="-O1 -g" SANITIZE="-fsanitize=thread" test-cpp

bench: $(BIN)/bench_lob $(BIN)/bench_spsc
	$(BIN)/bench_lob
	$(BIN)/bench_spsc

clean:
	rm -rf build

help:
	@sed -n '1,10p' Makefile

-include $(shell find $(BUILD)/obj -name '*.d' 2>/dev/null)
