# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /global/homes/c/chris_yt/final/sift-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /global/homes/c/chris_yt/final/sift-cpp/bulid

# Include any dependencies generated for this target.
include examples/CMakeFiles/match_features.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/match_features.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/match_features.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/match_features.dir/flags.make

examples/CMakeFiles/match_features.dir/match_features.cpp.o: examples/CMakeFiles/match_features.dir/flags.make
examples/CMakeFiles/match_features.dir/match_features.cpp.o: ../examples/match_features.cpp
examples/CMakeFiles/match_features.dir/match_features.cpp.o: examples/CMakeFiles/match_features.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/c/chris_yt/final/sift-cpp/bulid/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/match_features.dir/match_features.cpp.o"
	cd /global/homes/c/chris_yt/final/sift-cpp/bulid/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/match_features.dir/match_features.cpp.o -MF CMakeFiles/match_features.dir/match_features.cpp.o.d -o CMakeFiles/match_features.dir/match_features.cpp.o -c /global/homes/c/chris_yt/final/sift-cpp/examples/match_features.cpp

examples/CMakeFiles/match_features.dir/match_features.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/match_features.dir/match_features.cpp.i"
	cd /global/homes/c/chris_yt/final/sift-cpp/bulid/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /global/homes/c/chris_yt/final/sift-cpp/examples/match_features.cpp > CMakeFiles/match_features.dir/match_features.cpp.i

examples/CMakeFiles/match_features.dir/match_features.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/match_features.dir/match_features.cpp.s"
	cd /global/homes/c/chris_yt/final/sift-cpp/bulid/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /global/homes/c/chris_yt/final/sift-cpp/examples/match_features.cpp -o CMakeFiles/match_features.dir/match_features.cpp.s

# Object files for target match_features
match_features_OBJECTS = \
"CMakeFiles/match_features.dir/match_features.cpp.o"

# External object files for target match_features
match_features_EXTERNAL_OBJECTS =

../bin/match_features: examples/CMakeFiles/match_features.dir/match_features.cpp.o
../bin/match_features: examples/CMakeFiles/match_features.dir/build.make
../bin/match_features: src/libimg.a
../bin/match_features: src/libsift.a
../bin/match_features: src/libimg.a
../bin/match_features: src/libstb_image.a
../bin/match_features: examples/CMakeFiles/match_features.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/global/homes/c/chris_yt/final/sift-cpp/bulid/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/match_features"
	cd /global/homes/c/chris_yt/final/sift-cpp/bulid/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/match_features.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/match_features.dir/build: ../bin/match_features
.PHONY : examples/CMakeFiles/match_features.dir/build

examples/CMakeFiles/match_features.dir/clean:
	cd /global/homes/c/chris_yt/final/sift-cpp/bulid/examples && $(CMAKE_COMMAND) -P CMakeFiles/match_features.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/match_features.dir/clean

examples/CMakeFiles/match_features.dir/depend:
	cd /global/homes/c/chris_yt/final/sift-cpp/bulid && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/homes/c/chris_yt/final/sift-cpp /global/homes/c/chris_yt/final/sift-cpp/examples /global/homes/c/chris_yt/final/sift-cpp/bulid /global/homes/c/chris_yt/final/sift-cpp/bulid/examples /global/homes/c/chris_yt/final/sift-cpp/bulid/examples/CMakeFiles/match_features.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/match_features.dir/depend

