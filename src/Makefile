CXX 			:= g++
SRC 			:= $(wildcard ./*.cpp ./libs/*.cpp)
OBJ				:= $(patsubst %.cpp, %.o, $(SRC))
CXXFLAGS 		:= -std=c++11 -pipe -g
SHARED_LIBRARY 	=

# Add needed header path
SHARED_LIBRARY += -I/usr/local/cuda-11.4/include

# Add dynamic library
SHARED_LIBRARY += -pthread
SHARED_LIBRARY += -lonnxruntime
SHARED_LIBRARY += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn
# SHARED_LIBRARY += -L/usr/local/cuda/lib64 -lcuda -lcudart


all: $(OBJ)
	@echo Compiling main
	@$(CXX) $(CXXFLAGS) -o main $(OBJ) $(SHARED_LIBRARY)


%.o: %.cpp
	@echo Build $@
	@@$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: debug clean
debug:
	$(eval CXXFLAGS += -g)
	@:

clean:
	@find -name "*.o" -exec rm {} \;


