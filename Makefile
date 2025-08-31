CXX = g++
CXXFLAGS = -g -Wall -IHeaders

SRC_DIR = Source
TEST_DIR = Tests
OBJ_DIR = build

SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(TEST_DIR)/*.cpp)

OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(SRCS))

TARGET = val

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking"
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: %.cpp
	@echo "Compiling"
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)