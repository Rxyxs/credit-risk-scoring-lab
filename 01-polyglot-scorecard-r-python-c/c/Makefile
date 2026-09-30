# Compila el motor de scoring con GCC/Clang (Linux/macOS): una libreria
# compartida para que Python la cargue via ctypes, un ejecutable standalone
# para el benchmark en C puro, y el runner de pruebas unitarias.
#
# Ejecutar desde c/:
#   make            compila la libreria compartida y el benchmark
#   make test       compila y corre la suite de pruebas (exit code 0/1)
#   make clean
#
# Equivalente en Windows/MSVC: build.ps1 (produce outputs/models/score_engine.dll)

CC := gcc
CFLAGS := -std=c99 -Wall -Wextra -O3
LDLIBS := -lm

REPO_ROOT := ..
OUT_DIR := $(REPO_ROOT)/outputs/models
BUILD_DIR := build

UNAME_S := $(shell uname -s 2>/dev/null || echo Unknown)
ifeq ($(UNAME_S),Darwin)
  LIB_EXT := dylib
else
  LIB_EXT := so
endif

LIB := $(OUT_DIR)/libscore_engine.$(LIB_EXT)
BENCH := $(OUT_DIR)/score_bench
TEST_BIN := $(BUILD_DIR)/test_score_engine

.PHONY: all lib bench test clean

all: lib bench

lib: $(LIB)

bench: $(BENCH)

$(LIB): score_engine.c score_engine.h | $(OUT_DIR)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ score_engine.c $(LDLIBS)

$(BENCH): score_engine.c bench_main.c score_engine.h | $(OUT_DIR)
	$(CC) $(CFLAGS) -o $@ score_engine.c bench_main.c $(LDLIBS)

$(TEST_BIN): score_engine.c tests/test_score_engine.c score_engine.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ score_engine.c tests/test_score_engine.c $(LDLIBS)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -f $(LIB) $(BENCH) $(TEST_BIN)
	rm -rf $(BUILD_DIR)
