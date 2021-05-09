CC=g++
FLAGS=-g -O3 -std=c++17

.DEFAULT_GOAL:= fsstencil

fsstencil : 
	$(CC) $(FLAGS) -o fsstencil fu_semi.cpp
clean:
	rm fsstencil
