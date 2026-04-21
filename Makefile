CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall

# Targets
all: serial openmp

serial: src/serial/main.c src/serial/gmm.c
	mkdir -p bin
	$(CC) $(CFLAGS) $^ -lm -o bin/gmm_serial

openmp: src/openmp/main_omp.c src/openmp/gmm_omp.c
	mkdir -p bin
	$(CC) $(CFLAGS) -fopenmp $^ -lm -o bin/gmm_omp

clean:
	rm -rf bin/gmm_serial bin/gmm_omp
