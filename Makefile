PROJECTNAME = pca_icp
CPPFLAGS = -g -std=c++11 -Wall
LDFLAGS =
OBJDIR = obj/
SRCDIR = src/
INCDIRS = include/ /usr/include/eigen3/
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
