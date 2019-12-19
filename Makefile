default: cmake
# default: gcc

gcc:
	g++ -pthread -g2 -Wall -Wextra -std=c++11 main.cpp -ldl -o build/main
	./build/main

cmake:
	cd build && make
	./build/main

