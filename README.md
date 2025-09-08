# Design-and-Control-of-Cube-Drone
For more details on the controller running on a resource-constrained microcontroller, please refer to our project: [SM-NMPC](https://github.com/aralab-unr/SM-NMPC).  
It includes instructions on generating the NMPC code for the Teensy 4.1, as well as the setup used for failure scenarios in the system’s stability tests.

## Software Requirements & Setup

The simulation is configured with the following setup:
- Ubuntu 22.04
- ROS2 Humble
- Gazebo 11
- Xarco-ROS-Humble (sudo apt install ros-humble-xacro)
- Gazebo_ros_pkgs (sudo apt install ros-humble-gazebo-ros-pkgs)
- ACADO Toolkit (https://acado.github.io/)


Follow these commands in order to install the simulation of for the Cube-Drone:

```shell
# Step 1: Create and build a colcon workspace:
$ mkdir -p ~/ros2_ws/src
$ cd ~/ros2_ws/
$ colcon build
$ echo "source ~/ros2_ws/devel/setup.bash" >> ~/.bashrc

# Step 2: Clone this repo into your workspace
$ cd ~/ros2_ws/src
Download the folder smcmpcquad or the smcnmpccube in the main branch

# Step 3: Build the colcon workspace for this package
$ cd ~/ros2_ws
$ colcon build
```
* Note that the package contains the code generation and includes the qpOASES library. If the user wants to use SM-NMPC for a different problem, they need to regenerate the code and replace it to the include folder.
* Note that this project uses a custom plugin. Users need to replace the plugin path in the file /urdf/cube.urdf.xacro at line 1009. Replace: plugin name="cubeplugin" filename="/home/yourcomputername/dev_ws/install/smcnmpccube/lib/smcnmpccube/libcubeplugin.so" with the correct path by changing the username to the name of your computer. Then rebuild the project again to run the simulation.

## Simulation results


To run the simulation for Cube-Drone, follow these commands:

```shell
# Step 1: Run the Gazebo model:
$ ros2 launch smcnmpccube model.launch.py

# Step 2: Run the controller
$ ros2 run smcnmpccube smcnmpccube
```
## Motor failure results

To run the motor failure scenario, the user needs to modify the plugin in the /src/cubeplugin.cc. Change the added throttle to 50% as described in the manuscript. For the Cube, comment out line 177 and uncomment line 179.

Then rebuild the project and run the simulation as in the normal cases described above.
