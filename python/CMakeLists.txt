add_subdirectory(dune)

configure_file(setup.py.in setup.py)

dune_python_install_package(
  PATH "."
  CMAKE_METADATA_FILE dune/istl/metadata.cmake
  DEPENDS _istl
)
