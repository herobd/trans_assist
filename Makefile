
CXX = g++
LINK = g++
CXX_FLAGS_NO_G =  -std=c++11 -pthread 
#CXX_FLAGS_NO_G += -DOPENCV2 -DCPU_ONLY
CXX_FLAGS =  $(CXX_FLAGS_NO_G) -O3 #-g

LIBS += -lpthread

LIBS += -l:libopencv_core.so.3.1 -l:libopencv_imgcodecs.so.3.1 -l:libopencv_imgproc.so.3.1 -l:libopencv_highgui.so.3.1
LIBS += -lopencv_imgcodecs
LIBS += -pthread

INCLUDE += -I /usr/local/cuda/include


#LIBS += -L ../../brian_caffe/scripts/cnnspp_spotter -lcnnspp_spotter
#INCLUDE += -I ../../brian_caffe/scripts/cnnspp_spotter
#LIBS += -L ../../brian_caffe/build/lib
#INCLUDE +=-I ../../brian_caffe/include
#LIBS += -lcaffe -lboost_system

LIBS += -L ../caffe/scripts/cnnspp_spotter -lcnnspp_spotter
INCLUDE += -I ../caffe/scripts/cnnspp_spotter
LIBS += -L ../caffe/build/lib
INCLUDE +=-I ../caffe/include
LIBS += -lcaffe -l:libboost_system.so -lglog

PROGRAM_NAME = trans_assist


bin: $(PROGRAM_NAME)
all: $(PROGRAM_NAME)

clean:
	- rm  $(PROGRAM_NAME)

$(PROGRAM_NAME): trans_assist.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDE) -o $(PROGRAM_NAME) trans_assist.cpp $(LIBS) 
