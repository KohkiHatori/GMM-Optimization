CC = gcc
CFLAGS = -O3 -march=native -ffast-math -Wall

# Targets
all: serial

serial: src/serial/main.c src/serial/gmm.c
	mkdir -p bin
	$(CC) $(CFLAGS) $^ -lm -o bin/gmm_serial

clean:
	rm -rf bin/gmm_serial
