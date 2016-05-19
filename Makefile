PROJECTNAME = pca_icp
CPPFLAGS = -g -std=c++11
LDFLAGS =-lboost_system -lboost_thread -lpcl_common -lpcl_io -lpcl_visualization -lpcl_filters -lvtkCommonDataModel-6.0 -lvtkCommonCore-6.0 -lvtkCommonMath-6.0 -lvtkRenderingCore-6.0 -lvtkRenderingLOD-6.0
OBJDIR = obj/
SRCDIRS = src test
INCDIRS = include/ /usr/include/eigen3/ /usr/local/include/pcl-1.7/ /usr/include/vtk-6.0/ /usr/local/cuda/include/
INC = $(foreach d, $(INCDIRS), -I$d)
COMPILER = g++
CUDA_COMPILER = /usr/local/cuda/bin/nvcc
CUDA_INSTALL_PATH = /usr/local/cuda/
VPATH = src:test
DEPDIR := .d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

RM = rm -rf

SRCS = $(shell find $(SRCDIRS) -name "*.cpp" -o -name "*.cu")
SRCS1 = $(SRCS:.cu=.o)
OBJS = $(addprefix $(OBJDIR),$(SRCS1:.cpp=.o))

all: $(PROJECTNAME)

$(PROJECTNAME): $(OBJS)
	@echo Linking $<...
	@$(CUDA_COMPILER) $(OBJS) $(LDFLAGS) -o $(PROJECTNAME)


$(OBJDIR)%.o: %.cpp
$(OBJDIR)%.o: %.cpp $(DEPDIR)/%.d
	@echo Compiling $<...
	@mkdir -p $(@D)
	@mkdir -p $(DEPDIR)/$(*D)
	$(COMPILER) $(DEPFLAGS) $(CPPFLAGS) $(INC) -c $< -o $@
	@$(POSTCOMPILE)

$(OBJDIR)%.o: %.cu
#$(OBJDIR)%.o: %.cu $(DEPDIR)/%.d
	@echo Compiling $<...
	@mkdir -p $(@D)
	$(CUDA_COMPILER) $(CPPFLAGS) $(INC) -c $< -o $@
	$(CUDA_COMPILER) $(CPPFLAGS) $(INC) -E -Xcompiler "-isystem $(CUDA_INSTALL_PATH)/include -MM" $< -o $(DEPDIR)/$*.Td
	@$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))

clean:
		$(RM) $(OBJDIR)* $(PROJECTNAME) $(DEPDIR)/*

clear: clean $(PCA_ICP)
