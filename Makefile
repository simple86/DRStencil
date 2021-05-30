CC=g++
FLAGS=-g -O3 -std=c++17

.DEFAULT_GOAL:= drstencil

drstencil : 
	$(CC) $(FLAGS) -o drstencil main.cpp
clean:
	rm drstencil
