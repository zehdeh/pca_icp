PROJECTNAME = pca_icp
CPPFLAGS = -g -std=c++11 -Wall
LDFLAGS =-lboost_system -lboost_thread -lpcl_common -lpcl_io -lpcl_visualization -lpcl_filters -lvtkCommonDataModel-6.0 -lvtkCommonCore-6.0 -lvtkCommonMath-6.0 -lvtkRenderingCore-6.0 -lvtkRenderingLOD-6.0
OBJDIR = obj/
SRCDIR = src/
INCDIRS = include/ /usr/include/eigen3/ /usr/local/include/pcl-1.7/ /usr/include/vtk-6.0/
INC = $(foreach d, $(INCDIRS), -I$d)
COMPILER = g++

RM = rm -rf

SRCS = $(shell find $(SRCDIR) -name "*.cpp")
OBJS = $(addprefix $(OBJDIR),$(subst $(SRCDIR),,$(SRCS:.cpp=.o)))

all: $(PROJECTNAME)

$(PROJECTNAME): $(OBJS)
		$(COMPILER) $(OBJS) $(LDFLAGS) -o $(PROJECTNAME)

$(OBJDIR)%.o: $(SRCDIR)%.cpp
		@mkdir -p $(@D)
			$(COMPILER) $(CPPFLAGS) $(INC) -c $< -o $@

clean:
		$(RM) $(OBJDIR)* $(PROJECTNAME)

clear: clean $(PCA_ICP)
