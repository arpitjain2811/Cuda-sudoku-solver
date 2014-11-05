CC=nvcc
NVCCFLAGS=
OBJS= sudoku_solver

$(OBJS) : sudoku_solver.cu
				$(CC) $(NVCCFLAGS) sudoku_solver.cu	-o $(OBJS)

clean:
	rm -rf $(OBJS)