#Outdates and should not be used. Use CMAKE instead.
MFEM_DIR ?= /Users/patrikrac/Documents_local/bachelor/mfem/mfem

CC=g++
CFLAGS=-O3 -std=c++11 -I$(MFEM_DIR)

all: mfem_general

mfem_general : mfem_general.cpp
	$(CC) $(CFLAGS) $< -o $@ -L$(MFEM_DIR) -lmfem

clean: 
	rm mfem_general solution.vtk table.tex error_mfem.txt



#g++ -O3 -std=c++11 -I/Users/patrikrac/Documents_local/bachelor/mfem/mfem mfem_general.cpp -o mfem_general -L/Users/patrikrac/Documents_local/bachelor/mfem/mfem -lmfem
#g++ -O3 -std=c++11 -I../../../mfem/mfem -o mfem_general -L../../../mfem/mfem -lmfem
