cmake_minimum_required(VERSION 2.6)

project(ann2)
set( ANN2_SRC ann2.cc)

add_executable(example2 init2.cc ${ANN2_SRC} )
add_executable(example3 init3.cc ${ANN2_SRC} )
add_executable(example4 init4.cc ${ANN2_SRC} )
add_executable(example5 init5.cc ${ANN2_SRC} )
add_executable(example6 init6.cc ${ANN2_SRC} )

#
# Project (targets) definitions
#
add_definitions(-DNDEBUG) # -D_GLIBCXX_PARALLEL)


#
# Setup OpenMP
#
#find_package(OpenMP)
if(OPENMP_FOUND) 
  message(STATUS "Building with OpenMP support.")
endif()

#
# Setup Boost
#
set(Boost_COMPILER "-mgw45")
set(Boost_USE_STATIC_LIBS   ON )
set(Boost_USE_MULTITHREADED ON )
find_package( Boost 1.44.0 COMPONENTS program_options serialization)

if(Boost_FOUND)

  include_directories  (${Boost_INCLUDE_DIRS}   )
  include_directories  ("c:/libs/imglibs/libpng")
  include_directories  ("c:/libs/imglibs/zlib  ")

  set_property( TARGET example2 PROPERTY RUNTIME_OUTPUT_DIRECTORY "../bin/approx" )
  set_property( TARGET example3 PROPERTY RUNTIME_OUTPUT_DIRECTORY "../bin/recognition" )
  set_property( TARGET example4 PROPERTY RUNTIME_OUTPUT_DIRECTORY "../bin/cls/clusters" )
  set_property( TARGET example5 PROPERTY RUNTIME_OUTPUT_DIRECTORY "../bin/cls/clusters_par" )
  set_property( TARGET example6 PROPERTY RUNTIME_OUTPUT_DIRECTORY "../bin/cls/clusters_img" )

  target_link_libraries(example2 ${Boost_LIBRARIES}   )
  target_link_libraries(example3 ${Boost_LIBRARIES}   )
  target_link_libraries(example4 ${Boost_LIBRARIES}   )
  target_link_libraries(example5 ${Boost_LIBRARIES}   )

  find_library(LIB_PNG png "c:/libs/imglibs/libpng")
  target_link_libraries(example6 ${Boost_LIBRARIES} ${LIB_PNG}   )

#  message(FATAL_ERROR "Boost lobs: ${Boost_LIBRARIES}")
else()
  message(FATAL_ERROR "Boost library wasn't found!")
endif()



if(${CMAKE_COMPILER_IS_GNUCXX})
  set(PRJ_COMPILER_FLAGS "${COMPILE_FLAGS} ${OpenMP_CXX_FLAGS} -Wall -std=gnu++0x -s -O3 -march=i686 -ffast-math -march=native -mfpmath=sse -msse4" )
  set_target_properties(example2 PROPERTIES COMPILE_FLAGS ${PRJ_COMPILER_FLAGS} )
  set_target_properties(example3 PROPERTIES COMPILE_FLAGS ${PRJ_COMPILER_FLAGS} )
  set_target_properties(example4 PROPERTIES COMPILE_FLAGS ${PRJ_COMPILER_FLAGS} )
  set_target_properties(example5 PROPERTIES COMPILE_FLAGS ${PRJ_COMPILER_FLAGS} )
  set_target_properties(example6 PROPERTIES COMPILE_FLAGS ${PRJ_COMPILER_FLAGS} )

#  get_property(comp TARGET example2 PROPERTY COMPILE_FLAGS)
#  message(STATUS "Compile options: ${comp}")

  set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mconsole -Wl,--enable-auto-import -static-libstdc++ -static-libgcc" )
endif()

# TODO Compression of 'location'
if(CMAKE_HOST_WIN32)
  get_property(bin_name2 TARGET example2 PROPERTY LOCATION)
#  add_custom_command(TARGET example2 POST_BUILD
#	COMMAND "/opt/bin/upx.exe" "--best" ${bin_name2}
#	COMMENT "Compressing executable.")
endif()
