# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.15.5/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.15.5/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/mariannelado-roy/lentillos-nachos/calibration/tnm-opencv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/mariannelado-roy/lentillos-nachos/calibration/build

# Include any dependencies generated for this target.
include CMakeFiles/calibexeAV.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/calibexeAV.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/calibexeAV.dir/flags.make

CMakeFiles/calibexeAV.dir/demoAV.cc.o: CMakeFiles/calibexeAV.dir/flags.make
CMakeFiles/calibexeAV.dir/demoAV.cc.o: /Users/mariannelado-roy/lentillos-nachos/calibration/tnm-opencv/demoAV.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/mariannelado-roy/lentillos-nachos/calibration/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/calibexeAV.dir/demoAV.cc.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/calibexeAV.dir/demoAV.cc.o -c /Users/mariannelado-roy/lentillos-nachos/calibration/tnm-opencv/demoAV.cc

CMakeFiles/calibexeAV.dir/demoAV.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/calibexeAV.dir/demoAV.cc.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mariannelado-roy/lentillos-nachos/calibration/tnm-opencv/demoAV.cc > CMakeFiles/calibexeAV.dir/demoAV.cc.i

CMakeFiles/calibexeAV.dir/demoAV.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/calibexeAV.dir/demoAV.cc.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mariannelado-roy/lentillos-nachos/calibration/tnm-opencv/demoAV.cc -o CMakeFiles/calibexeAV.dir/demoAV.cc.s

# Object files for target calibexeAV
calibexeAV_OBJECTS = \
"CMakeFiles/calibexeAV.dir/demoAV.cc.o"

# External object files for target calibexeAV
calibexeAV_EXTERNAL_OBJECTS =

calibexeAV: CMakeFiles/calibexeAV.dir/demoAV.cc.o
calibexeAV: CMakeFiles/calibexeAV.dir/build.make
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_stitching.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_superres.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_videostab.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_aruco.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_bgsegm.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_bioinspired.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_ccalib.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_dnn_objdetect.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_dpm.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_face.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_freetype.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_fuzzy.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_hdf.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_hfs.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_img_hash.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_line_descriptor.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_optflow.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_reg.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_rgbd.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_saliency.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_stereo.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_structured_light.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_surface_matching.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_tracking.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_xfeatures2d.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_ximgproc.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_xobjdetect.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_xphoto.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_shape.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_photo.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_calib3d.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_phase_unwrapping.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_video.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_datasets.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_plot.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_text.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_dnn.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_features2d.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_flann.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_highgui.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_ml.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_videoio.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_imgcodecs.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_objdetect.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_imgproc.3.4.2.dylib
calibexeAV: /Users/mariannelado-roy/miniconda3/lib/libopencv_core.3.4.2.dylib
calibexeAV: CMakeFiles/calibexeAV.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/mariannelado-roy/lentillos-nachos/calibration/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable calibexeAV"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/calibexeAV.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/calibexeAV.dir/build: calibexeAV

.PHONY : CMakeFiles/calibexeAV.dir/build

CMakeFiles/calibexeAV.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/calibexeAV.dir/cmake_clean.cmake
.PHONY : CMakeFiles/calibexeAV.dir/clean

CMakeFiles/calibexeAV.dir/depend:
	cd /Users/mariannelado-roy/lentillos-nachos/calibration/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/mariannelado-roy/lentillos-nachos/calibration/tnm-opencv /Users/mariannelado-roy/lentillos-nachos/calibration/tnm-opencv /Users/mariannelado-roy/lentillos-nachos/calibration/build /Users/mariannelado-roy/lentillos-nachos/calibration/build /Users/mariannelado-roy/lentillos-nachos/calibration/build/CMakeFiles/calibexeAV.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/calibexeAV.dir/depend

