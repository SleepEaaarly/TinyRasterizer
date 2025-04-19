SYSCONF_LINK = nvcc
CPPFLAGS     =
LDFLAGS      = -lgdi32
LIBS         = -lcudart -luser32  # 添加 CUDA 库

DESTDIR = ./
TARGET  = main

# 获取所有 .cpp 和 .cu 文件
CPP_SOURCES := $(wildcard *.cpp)
CU_SOURCES  := $(wildcard *.cu)

# 生成对应的 .o 文件
CPP_OBJECTS := $(patsubst %.cpp,%.o,$(CPP_SOURCES))
CU_OBJECTS  := $(patsubst %.cu,%.o,$(CU_SOURCES))
OBJECTS     := $(CPP_OBJECTS) $(CU_OBJECTS)

all: $(DESTDIR)$(TARGET)

$(DESTDIR)$(TARGET): $(OBJECTS)
	$(SYSCONF_LINK) -o $(DESTDIR)$(TARGET) $(OBJECTS) $(LIBS) $(LDFLAGS)

# 编译 .cpp 文件的规则
$(CPP_OBJECTS): %.o: %.cpp
	$(SYSCONF_LINK) $(CPPFLAGS) -c $< -o $@ $(LDFLAGS)

# 编译 .cu 文件的规则
$(CU_OBJECTS): %.o: %.cu
	$(SYSCONF_LINK) $(CPPFLAGS) -c $< -o $@ $(LDFLAGS)

run: $(DESTDIR)$(TARGET)
	$(DESTDIR)$(TARGET)

clean:
	-rm -f $(OBJECTS)
	-rm -f $(TARGET)
	-rm -f rst/*.png