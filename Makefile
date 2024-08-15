CUDA_PATH ?= /usr/local/cuda

cuda_override: cuda_override.cpp
	g++ -fPIC -DPIC -c cuda_override.cpp -I$(CUDA_PATH)/include -fno-use-cxa-atexit -std=c++20 -O3
	ld -shared -o cuda_override.so cuda_override.o -ldl

cannyEdgeDetectorNPP/cannyEdgeDetectorNPP: cannyEdgeDetectorNPP/cannyEdgeDetectorNPP.cpp
	cd cannyEdgeDetectorNPP && $(MAKE)

build: cuda_override cannyEdgeDetectorNPP/cannyEdgeDetectorNPP

run: build
	nu run.nu
