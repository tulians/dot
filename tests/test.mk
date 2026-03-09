# Makefile for host tests

CC=gcc
CFLAGS=-Ilibs/unity/inc -I$(PROGRAM_PATH_AND_NAME)/inc -Wall -Wextra
TEST_SRC=libs/unity/src/unity.c tests/src/test_logic.c $(PROGRAM_PATH_AND_NAME)/src/logic.c $(PROGRAM_PATH_AND_NAME)/src/benchmark.c

test:
	@echo Building and running host tests...
	$(CC) $(CFLAGS) $(TEST_SRC) -o tests/runner
	./tests/runner
	@rm tests/runner
