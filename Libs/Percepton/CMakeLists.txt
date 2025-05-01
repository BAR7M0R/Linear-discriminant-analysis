add_library(Percepton STATIC Src/Percepton.cpp)
target_include_directories(Percepton PRIVATE Inc)
target_compile_features(Percepton PUBLIC cxx_std_20)
target_link_libraries(Percepton PRIVATE Matrix_tools)


