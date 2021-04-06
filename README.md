### Install python dependencies

```bash
pip install -r requirements.txt
```

### Running
```bash
# Run as ego 1
iac_planner $ python run_me.py
iac_planner $ python run_me.py 1

# Run as other egos
iac_planner $ python run_me.py 2
iac_planner $ python run_me.py 5
# Disable plotting
# Can also be done by changing ./iac_planner/helpers.py
iac_planner $ python run_me.py 1 --no-plot
iac_planner $ python run_me.py 2 --no-plot
iac_planner $ python run_me.py 5 --no-plot

# Currently argument parsing is very rudimentary and only supports the above forms
```

requires numpy, scipy, pandas, rticonnextdds-connector, matplotlib [optional]

For [Team-Abhiyaan](http://github.com/Team-Abhiyaan/)

uses [spidyadi/Collision_Checker](https://github.com/spidyadi/Collision_Checker)

Extends [surajRathi/path_score](http://github.com/surajRathi/path_score)