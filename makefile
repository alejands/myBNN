train: net.o train.o
	g++ -o train net.o train.o

train.o: train.cpp net.h
	g++ -c train.cpp net.h

net.o: net.h
	g++ -c net.cpp net.h

clean:
	rm *.o train
