MFEM_DIR ?= /Users/patrikrac/Documents_local/Dateien/programms/bachelor/mfem/mfem-3.4

CC=g++
CFLAGS=-O3 -std=c++11 -I$(MFEM_DIR)

all: mfem_general

mfem_general : mfem_general.cpp
	$(CC) $(CFLAGS) $< -o $@ -L$(MFEM_DIR) -lmfem

clean: 
	rm mfem_general solution.vtk table.tex errorMFEM.txt



#g++ -O3 -std=c++11 -I/Users/patrikrac/Documents_local/Dateien/programms/bachelor/mfem/mfem-3.4 mfem_general.cpp -o mfem_general -L/Users/patrikrac/Documents_local/Dateien/programms/bachelor/mfem/mfem-3.4 -lmfem
#g++ -O3 -std=c++11 -I../../../mfem/mfem-3.4 -o mfem_general -L../../../mfem/mfem-3.4 -lmfem
