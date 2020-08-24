# Defines the functions to use SuperLU
#
# .. cmake_function:: add_dune_superlu_flags
#
#    .. cmake_param:: targets
#       :positional:
#       :single:
#       :required:
#
#       A list of targets to use SuperLU with.
#

include_guard(GLOBAL)

# set HAVE_SUPERLU for config.h
set(HAVE_SUPERLU ${SUPERLU_FOUND})

function(add_dune_superlu_flags)
  if(SUPERLU_FOUND)
    cmake_parse_arguments(_add_superlu "OBJECT" "" "" ${ARGN})
    foreach(_target ${_add_superlu_UNPARSED_ARGUMENTS})
      if(NOT _add_superlu_OBJECT)
        target_link_libraries(${_target} PUBLIC ${SUPERLU_DUNE_LIBRARIES})
      endif()
      target_include_directories(${_target} PUBLIC ${SUPERLU_INCLUDE_DIRS})
      target_compile_definitions(${_target} PUBLIC ENABLE_SUPERLU=1)
    endforeach()
  endif(SUPERLU_FOUND)
endfunction(add_dune_superlu_flags)
