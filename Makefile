cuda_override: cuda_override.cpp
	g++ -fPIC -DPIC -c cuda_override.cpp -I/opt/cuda/include -fno-use-cxa-atexit -std=c++20
	ld -shared -o cuda_override.so cuda_override.o -ldl

cannyEdgeDetectorNPP/cannyEdgeDetectorNPP: cannyEdgeDetectorNPP/cannyEdgeDetectorNPP.cpp
	cd cannyEdgeDetectorNPP && $(MAKE)

run: cuda_override cannyEdgeDetectorNPP/cannyEdgeDetectorNPP
	nu run.nu
