PROJECTNAME = pca_icp
CPPFLAGS = -g -std=c++11
LDFLAGS =-lboost_system -lboost_thread -lpcl_common -lpcl_io -lpcl_visualization -lpcl_filters -lvtkCommonDataModel-6.0 -lvtkCommonCore-6.0 -lvtkCommonMath-6.0 -lvtkRenderingCore-6.0 -lvtkRenderingLOD-6.0
OBJDIR = obj/
SRCDIRS = src test
INCDIRS = include/ /usr/include/eigen3/ /usr/local/include/pcl-1.7/ /usr/include/vtk-6.0/ /usr/local/cuda/include/
INC = $(foreach d, $(INCDIRS), -I$d)
COMPILER = /usr/local/cuda/bin/nvcc
VPATH = src:test

RM = rm -rf

SRCS = $(shell find $(SRCDIRS) -name "*.cpp" -o -name "*.cu")
SRCS1 = $(SRCS:.cu=.o)
OBJS = $(addprefix $(OBJDIR),$(SRCS1:.cpp=.o))

all: $(PROJECTNAME)

$(PROJECTNAME): $(OBJS)
		$(COMPILER) $(OBJS) $(LDFLAGS) -o $(PROJECTNAME)

$(OBJDIR)%.o: %.cpp
		@mkdir -p $(@D)
			$(COMPILER) $(CPPFLAGS) $(INC) -c $< -o $@

$(OBJDIR)%.o: %.cu
		@mkdir -p $(@D)
			$(COMPILER) $(CPPFLAGS) $(INC) -c $< -o $@

clean:
		$(RM) $(OBJDIR)* $(PROJECTNAME)

clear: clean $(PCA_ICP)
