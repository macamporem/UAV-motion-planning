## Project: 3D Motion Planning

Miguel Campo | macamporem@gmail.com | 6036673655



![Quad Image](./misc/enroute_probabilistic.png)

---



### Explain the Starter Code

#### 1. Explain the functionality of what's provided in `motion_planning.py` and `planning_utils.py`
The file **planning_utils.py** implements the functions to create 2d grid representations of 3 dimensional volumes based on target altitude, and to perform 2d optimal path search in the 2d grid based on Euclidean heuristic penalization and unitary action cost in the four directions (north, south, east, west).  

The file **motion_planning.py** implements the events driven framework to determine and execute the flight plan.  The code defines the sequence of possible states the vehicle may be in (MANUAL, ARMING...), as well as three possible callback pathways (location, velocity and state).  The logic in the code orchestrates the transitions between states.  Inmediately after the vehicle is armed, the following sequence takes place: the vehicle state switches to PLANNING, the function plan_path() creates the list of waypoints, the vehicle takes off (through a state callback), the state swtches to WAYPOINT (through a local poition callback), and the vehicle starts to fly towars the sequence of waypoints (again, through local position callbacks) until it has empited the waypoints list.  When this happens, the state swtiches to LANDING, and the vehicle lands and disarms (through a velocity callback).

### Implementing Your Path Planning Algorithm

#### 1. Set your global home position
**Line 130-135 in motion_planning.py** read the first line of the csv file, extract lat0 and lon0 as floating point values and use the self.set_home_position() method to set global home.  The code uses *str.split* method and a list comprenhension to populate the values of *lat0* and *lan0*.

#### 2. Set your current local position
**Line 139 in motion_planning.py** determines the quadruptor's local position relative to global home using the *global_to_local* method and the *global_position* coordinates.

#### 3. Set grid start position from local position
**Line 152-155 in motion_planning.py** compute the start location in grid N-E coordinates by shifting the NED start coordinates to grid coordinates.

#### 4. Set grid goal position from geodetic coords
**Line 160-164 in motion_planning.py** compute the end (goal) location in grid N-E coordinates by transforming goal lat/lon global coordinates into NED coordinates, and by shifting the NED coordinates to grid coordinates.

#### 5. Modify A* to include diagonal motion (or replace A* altogether)
**Lines 99-106 and 128-144 in planning_utils.py** update the A* implementation to include diagonal motions on the grid with action cost of sqrt(2).  In **Lines 99-106**, the *deltas* of diagonal directions are simply defined as the addition of the deltas of N/S and E/W directions.  In **Lines 128-144**, we define the conditions for inclusion of the diagonal directions in the set of valid actions.  If a diagonal direction collides with an obstacle or if it leads to a point out of the grid, the diagonal direction is eliminated from the list of valid actions.

#### 6. Cull waypoints 
**Line 207 in motion_planning.py and 27-39 in planning_utils.py** eliminate the redundant waypoints from the optimal flight plan.  **Lines 27-39 in planning_utils.py** implement a recursive function to eliminate the redundant waypoints.  

The recursive function explores the first three points in the list of waypoints.  If they are colinear or more or less aligned (we experiment with different values of ceiling for the determinant), then the third point is eliminated.  The colinear function is then applied again over the remaining list of waypoints until there are only three waypoints left.  

# Extra Challenges: 

<u>**Heading**</u>.  **Lines 199-202 in motion_planning.py** contain the logic to set the heading of the waypoints in alignment with the (planned) direction of movement.

**<u>Optimal Plan</u>**.  The A* function works well for small distances.  For larger distances, the code can raise a time out error or take too long to compute.

The code implements two other optimization methods

- Medial Axis (path can be wobbly)
- **Probabilistic Planning (default method)**

To select which model to use, **Lines 174-196 in motion_planning.py** contain the logic to optimize using A* or any other other two models.

With regard to the two graph optimization models provided:

- **Lines 201-247 in planning_utils.py** implement a modified version of A* that can be used with the two graph methodologies above (Medial Axis and Probabilistic Planning)
- **Lines 252-307 in planning_utils.py** implement the logic to draw grid points from a random distribution for Probabilistic Planning.  The code keeps those points that are outside of the obstacles, and returns the following 3 objects: 
  - the list of points
  - the list of obstacles (polygons)
  - the list of polygon centroids
- **Lines 320-376 in planning_utils.py** implement the algorithm for Probabilistic Planning.  The algorithm uses Polygon class functions to determine which possible routes intersect with obstacles.  Those functions are very time consuming.  In order to accelerate the solution, the code uses the polygon centroids to pre-select first a shorter list of risky obstables, and then checks possible intersection with the obstacles in the pre-selected risky list.  
- **Lines 245-270 in motion_planning.py** compute the list of Probabilist optimal waypoints *<u>before</u>* the quadruptor is armed.  This is done to avoid time out errors that would emerge if we try to optimize after arming the quadruptor.
- **Lines 378-414 in planning_utils.py** implement the logic of Medial Axis path optimization without modification.

![Quad Image](./misc/enroute_probabilistic_2.png)

![Quad Image](./misc/enroute_probabilistic_3.png)

![Quad Image](./misc/enroute_over_trees.png)