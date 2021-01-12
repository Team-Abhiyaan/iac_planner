### Install python dependencies

```bash
pip install -r requirements.txt
```

```bash
colcon build --symlink-install --packages-up-to iac_planner
ros2 launch iac_planner main.launch.py
```

requires colcon, ros2, rviz2, numpy, and scipy.spatial

For [Team-Abhiyaan](http://github.com/Team-Abhiyaan/)

uses [spidyadi/Collision_Checker](https://github.com/spidyadi/Collision_Checker)

Extends [surajRathi/path_score](http://github.com/surajRathi/path_score)