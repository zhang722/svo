SVO
===

Disclaimer
----------

SVO has been tested under ROS Groovy and Hydro and Ubuntu 12.04 and 13.04. This is research code, expect that it changes often and any fitness for a particular purpose is disclaimed.

Installation Instructions
-------------------------

#### OPTIONAL: g2o - General Graph Optimization

Only required if you want to run bundle adjustment. It is not necessary for visual odometry.
g2o requires the following system dependencies: `cmake, libeigen3-dev, libsuitesparse-dev, libqt4-dev, qt4-qmake, libqglviewer-qt4-dev`, install them with `apt-get`
    
I suggest an out-of-source build of g2o:

    cd workspace
    git clone https://github.com/RainerKuemmerle/g2o.git
    cd g2o
    mkdir build
    cd build
    cmake ..
    make
    sudo make install

If you don't want to make a system install, then you can replace the cmake command with `cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME/installdir` 

#### Sophus - Lie groups

Sophus by Hauke Strasdat implements Lie groups that we need to describe rigid body transformations.

    cd workspace
    git clone https://github.com/strasdat/Sophus.git
    cd Sophus
    git checkout a621ff
    mkdir build
    cd build
    cmake ..
    make

You don't need to install the library since `cmake ..` writes the package location to `~.cmake/packages/` where CMake can later find it.

#### Fast Detector

The Fast detector by Edward Rosten is used to detect corners.
To simplify installation we provide a CMake package that contains the fast detector from the libCVD library (http://www.edwardrosten.com/cvd/).

    cd workspace
    git clone https://github.com/uzh-rpg/fast.git
    cd fast
    mkdir build
    cd build
    cmake ..
    make

#### ViKit - Some useful tools that we need

ViKit for instance contains camera models, some math and interpolation functions that SVO needs.
ViKit is a catkin project, therefore, download it into your catkin workspace source folder.

    cd catkin_ws/src
    git clone https://github.com/uzh-rpg/rpg_vikit.git

#### SVO

Now we are ready to build SVO.
Clone it into your catkin workspace

    cd catkin_ws/src
    git clone https://github.com/uzh-rpg/rpg_slam.git

If you installed g2o then set `HAVE_G2O` in `svo/CMakeLists.txt` to TRUE.
Then build

    catkin_make


Run SVO on a Dataset
-------------------------

Download this example dataset: [rpg.ifi.uzh.ch/datasets/airground_rig_s3_2013-03-18_21-38-48.bag](http://rpg.ifi.uzh.ch/datasets/airground_rig_s3_2013-03-18_21-38-48.bag)

Open a new console and start SVO with the prepared launchfile:

    roslaunch svo_ros test_rig3.launch
    
Open a new console and start RViz

    rosrun rviz rviz
    
In RViz, load the configuration file (File > Open Config) which is stored in `svo_ros/rviz_config.rviz`.

Now you are ready to start the rosbag. Open a new console and change to the directory where you have downloaded the example dataset. Then type:

    rosbag play airground_rig_s3_2013-03-18_21-38-48.bag
    
Now you should see the video with tracked features (green) and in RViz how the camera moves. 

SVO GUI
-------

Type `rosrun rqt_svo rqt_svo` to run the SVO widget that displays the number of tracked features, the frame rate and provides some interface buttons.

Keyboard Shortcuts
------------------

Make sure to active the console window when pressing the keys.

* `s`   Start/Restart
* `q`   Quit
* `r`   Reset

Parameter Settings
------------------

A description of all parameters which can be set via the launchfile is provided in `svo/include/config.h`. The default parameters can be viewed in `svo/src/config.cpp`. Moreover, some additional parameters (mainly rostopic names etc.) are read from the ros parameter server in `svo_ros/slam_node.cpp`.

Generating Code Documentation
-----------------------------

You can generate a Doxygen documentation as follows

    cd svo
    doxygen Doxyfile

Contributing
------------

You are very welcome to contribute by opening a pull request via Github.
I try to follow the ROS C++ style guide.

Licence
-------

The source code is released under GPLv3 licence. A professional edition license for closed-source projects is also available. Please contact `forster at ifi dot uzh dot ch` for further information.

Citing
------

If you use SVO in an academic context, please cite the following publication:

    @article{Pomerleau12comp,
      author = {Forster, Christian and Pizzoli, Matia and Scaramuzza, Davide},
      title = {{SVO: Fast Semi-Direct Monocular Visual Odometry}},
      journal = {IEEE International Conference on Robotics and Automation (ICRA)},
      year = {2014}
    }