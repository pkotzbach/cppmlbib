CXX = g++
CXXFLAGS = -g -Wall -IHeaders

SRC_DIR = Source
TEST_DIR = Tests
EX_DIR = Examples
OBJ_DIR = build

# Common sources (no main files here)
COMMON_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
COMMON_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(COMMON_SRCS))

# Specific sources
TEST_MAIN = $(TEST_DIR)/tests.cpp
IRIS_MAIN = $(EX_DIR)/iris.cpp

TEST_OBJ = $(OBJ_DIR)/tests.o
IRIS_OBJ = $(OBJ_DIR)/iris.o

# Targets
TEST_TARGET = tests
IRIS_TARGET = iris

all: $(TEST_TARGET) $(IRIS_TARGET)

# Build tests
$(TEST_TARGET): $(COMMON_OBJS) $(TEST_OBJ)
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) -o $@ $^

# Build iris
$(IRIS_TARGET): $(COMMON_OBJS) $(IRIS_OBJ)
	@echo "Linking $@"
	$(CXX) $(CXXFLAGS) -o $@ $^

# Generic compilation rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<"
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	@echo "Compiling $<"
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(EX_DIR)/%.cpp
	@echo "Compiling $<"
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TEST_TARGET) $(IRIS_TARGET)
