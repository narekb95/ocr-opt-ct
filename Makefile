release: main.cpp
	g++ -std=c++14 -O2 -o release main.cpp

out: main.cpp
	g++ -D__DEBUG -std=c++14 -Wall -Werror -g -O0 -o out main.cpp

run: release
	./scripts/test_solution.py

frelease: main.cpp
	g++ -std=c++14 -O2 -D__FILEIO -o fre main.cpp

lite: main.cpp
	g++ -std=c++14 -D__LITE -O2 -o lite main.cpp