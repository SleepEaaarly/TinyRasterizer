SYSCONF_LINK = g++
CPPFLAGS     =
LDFLAGS      = -lgdi32
LIBS         = -lm

DESTDIR = ./
TARGET  = main

OBJECTS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: $(DESTDIR)$(TARGET)

$(DESTDIR)$(TARGET): $(OBJECTS)
	$(SYSCONF_LINK) -Wall -o $(DESTDIR)$(TARGET) $(OBJECTS) $(LIBS) $(LDFLAGS)

$(OBJECTS): %.o: %.cpp
	$(SYSCONF_LINK) -Wall $(CPPFLAGS) -c $(CFLAGS) $< -o $@ $(LDFLAGS)

run: $(DESTDIR)$(TARGET)
	$(DESTDIR)$(TARGET)

clean:
	-rm -f $(OBJECTS)
	-rm -f $(TARGET)
	-rm -f *.tga

