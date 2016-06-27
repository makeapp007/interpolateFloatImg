all:  interpolateFloatImg 
	

interpolateFloatImg: interpolateFloatImg.cpp
	g++ -Ofast -ffast-math -march=native -flto -fwhole-program  -std=c++11 -fopenmp -o interpolateFloatImg interpolateFloatImg.cpp

