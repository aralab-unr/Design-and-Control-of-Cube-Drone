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
Download the folder nmpcsmccube in the main branch

# Step 3: Build the colcon workspace for this package
$ cd ~/ros2_ws
$ colcon build
```
* Note that the package contains the code generation and includes the qpOASES library. If the user wants to use SM-NMPC for a different problem, they need to regenerate the code and replace it to the include folder.
* Note that this project uses a custom plugin. Users need to replace the plugin path in the file /urdf/cube.urdf.xacro at line 1132. Replace: plugin name="cubeplugin" filename="/home/yourcomputername/dev_ws/install/smcnmpccube/lib/smcnmpccube/libcubeplugin.so" with the correct path by changing the username to the name of your computer. Then rebuild the project again to run the simulation.

## Simulation results


To run the simulation for Cube-Drone, follow these commands:

```shell
# Step 1: Run the Gazebo model:
$ ros2 launch nmpcsmccube wall.launch.py

# Step 2: Run the trajectory
$ ros2 run nmpcsmccube lidartrajectoryceiling
or
$ ros2 run nmpcsmccube lidartrajectorywall

# Step 3: Run the controller
$ ros2 run nmpcsmccube nmpcsmccube

```shell

## Motor failure results

To run the motor failure scenario, the user needs to modify the plugin in the /src/cubeplugin.cc. Change the added throttle to 50% as described in the manuscript. For the Cube, comment out line 176 and uncomment line 179.

Then rebuild the project and run the simulation as in the normal cases described above.

# Experiments
## Hardware Setups
We used the following sensors to control the Cube-Drone:
- Teensy 4.1 
- MPU 6050
- Micoair MTF-02P optical flow and LiDAR sensor (MTF-02P), TF Mini 1.8.1 for inspection
- FrSky X8R radio receiver

We used the following hardware to collect the data for 3D mapping and steel inspection:
- NVIDIA Jetson Orin
- Velodyne VLP-16 Puck LiDAR Sensor
- ZED Mini Stereo Camera for image capture
- DC HOUSE Mini Electric Linear Actuator Stroke
- Ultrasonic Thickness Gauge
  
## Embedded library
To embed the controller onto the Teensy 4.1, the user must download the ArduinoSMNMPC folder. This folder contains the library for NMPC (Nonlinear Model Predictive Control) and the implemented code for the flying robot. The user must copy the code generation library into the libraries folder of the Arduino IDE and then upload the sketch nmpcpidteensy.ino to the Teensy 4.1. As discussed earlier, if the user wants to run their own NMPC controller for another system, they must generate the code according to the specific problem and then replace all the generated files in the library to update the embedded code. (Note that qpOASES is already included in the library.)

* Note: The embedded code may have some missing includes in the acado_qpoases_interface.cpp file. To fix this, change #include "INCLUDE/EXTRAS/SolutionAnalysis.hpp" to #include "SolutionAnalysis.hpp" and change #include "INCLUDE/QProblemB.hpp" to #include "QProblemB.hpp".
  
## Results
### Stability test
To evaluate the stability of the Cube-Drone, we conducted tests on motor failure scenarios as well as its ability to fly through a narrow culvert. The hardware setup for the motor failure case is shown below:
<p align="center">
  <img src="figures/failurehardware.png" alt="failure" width="750">
</p>
<p align="center">
  <img src="figures/performancecube.png" alt="performacne" width="750">
</p>
<p align="center">
  <img src="figures/pwmcube.png" alt="pwmnormanl" width="400">
  <img src="figures/pwmcubefailure.png" alt="pwmfailure" width="400">
</p>
The Cube-Drone flew through a narrow culvert:
<p align="center">
  <img src="figures/flyculvert3d.png" alt="flyculvert" width="900">
</p>
<p align="center">
  <img src="figures/imuculvertangular.png" alt="imuculvert" width="400">
  <img src="figures/imuculvertacc.png" alt="accculvert" width="400">
</p>

### Multifunctional working mode test

We further evaluated the Cube-Drone’s capabilities in climbing walls, approaching ceilings, and swimming with added buoyancy.

<p align="center">
  <img src="figures/swimming.png" alt="swimming" width="450">
  <img src="figures/imuswimming.png" alt="imuswimming" width="350">
</p>
<p align="center">
  <img src="figures/ceilling.png" alt="ceilling" width="450">
  <img src="figures/imucelling.png" alt="imucelling" width="350">
</p>
<p align="center">
  <img src="figures/ground.png" alt="ground" width="450">
  <img src="figures/imuground.png" alt="imuground" width="350">
</p>
We also evaluated the Cube-Drone’s ability to transform from ground-running mode to flying mode in order to pass a stair, demonstrating the system’s adaptability to different terrain environments
<p align="center">
  <img src="figures/stair1.png" alt="stair1" width="800">
</p>
<p align="center">
  <img src="figures/orientationstair.png" alt="orientationstair" width="350">
  <img src="figures/positionstair.png" alt="positionstair" width="350">
</p>

### Data collection and mapping
To use the SLAM algorithm, follow the command:
```shell
$ ros2 launch lio_imm liom.launch.py \pointcloud_topic:=<\cloud topic> \imu_topic:=<\imu topic> \img_topic:=<\img topic>
```
<p align="center">
  <img src="figures/SLAM_results.png" alt="failure" width="900">
</p>

<p align="center">
  <img src="figures/slamcompare.png" alt="slamcompare" width="900">
</p>

### Inpsection
To validate this capability, we conducted experiments in three different locations:  
-Indoor inspection  
-Outdoor inspection  
-Closed-loop inspection on the campus bridge  
The data from the indoor inspection is as follows:
<p align="center">
  <img src="figures/indoorinspectioncube.png" alt="indoorinspectioncube" width="800">
</p>
<p align="center">
  <img src="figures/dataindoorinspection.png" alt="dataindoorinspectioncube" width="400">
</p>
The figure of outdoor inspection:
<p align="center">
  <img src="figures/outdoorinspectioncube.png" alt="outdoorinspectioncube" width="800">
</p>
The figure of bridge inspection:
<p align="center">
  <img src="figures/bridgeinspectioncube.png" alt="bridgeinspectioncube" width="800">
</p>
