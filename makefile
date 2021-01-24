PROG = imgp
CC = g++
CPPFLAGS = -g -Wall -I"include" -I"$(OCL_INCLUDE)"
LDFLAGS = -l OpenCl -L"$(OCL_LIB)" -L"lib" -l FreeImage -O2 -lgdi32
OBJS = Main.o

$(PROG) : $(OBJS)
	$(CC) -o $(PROG) $(OBJS) $(LDFLAGS)

Main.o :
	$(CC) $(CPPFLAGS) -c Main.cpp

clean:
	rm -f $(OBJS)