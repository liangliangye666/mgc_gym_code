cmake_minimum_required(VERSION 3.5)

project(y4a_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Num.msg"
  "msg/AllJointStates.msg"
  "msg/AllJointWithBaseStates.msg"
  "msg/Double.msg"
  "msg/Vector2d.msg"
  "msg/Vector3d.msg"
  "msg/Vector4d.msg"
  "msg/Vector8d.msg"
  "msg/Vector10d.msg"
  "msg/Vector14d.msg"
  "msg/Vector16d.msg"
)
ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif() 

set(BASE_DIR ${CMAKE_SOURCE_DIR})
file(RELATIVE_PATH RELATIVE_PATH_TO_BASE ${BASE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES xml/l4aurdf20241119.xml
        DESTINATION ${RELATIVE_PATH_TO_BASE}/xml)

install(FILES urdf/l4aurdf20241119.urdf
        DESTINATION ${RELATIVE_PATH_TO_BASE}/urdf)
        
if(SIM_ENABLE)
  install(DIRECTORY meshes
          DESTINATION ${RELATIVE_PATH_TO_BASE})
endif()
ament_package()
