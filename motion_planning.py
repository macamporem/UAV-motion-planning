import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import a_star, heuristic, create_grid, \
    colinear_reduction, test_path, optimal_probabilistic, \
    optimal_medial_axis
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection, dest_global_pos=None, mode='medial_axis', probabilistic_path=None):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.check_state = {}
        self.dest_global_pos = dest_global_pos
        self.mode = mode
        self.TARGET_ALTITUDE = int(self.dest_global_pos[2])
        self.SAFETY_DISTANCE = 5
        self.target_position[2] = self.TARGET_ALTITUDE
        if self.mode == 'probabilistic':
            self.probabilistic_path = probabilistic_path
        self.in_mission = True

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = self.TARGET_ALTITUDE
        SAFETY_DISTANCE = self.SAFETY_DISTANCE

        # DONE: read lat0, lon0 from colliders into floating point values
        with open('colliders.csv') as f:
            lat0, lon0 = [float(s.split(' ')[1]) for s in str(f.readline()).split(', ')]
        print(lon0,lat0)

        # DONE: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0,lat0,0)        

        # DONE: retrieve current global position
        # DONE: convert to current local position using global_to_local()
        local_position_ned = global_to_local(self.global_position, self.global_home)
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        #grid_start = (-north_offset, -east_offset)
        # DONE: convert start position to current position rather than map center
        # this is the point of the SF grid where the drone is located
        local_start_delta =  \
                [local_position_ned[0] - north_offset,
                local_position_ned[1] - east_offset]
        grid_start = tuple([int(np.ceil(x)) for x in local_start_delta])

        # Set goal as some arbitrary position on the grid
        #grid_goal = (-north_offset + 10, -east_offset + 10)
        # DONE: adapt to set goal as latitude / longitude position and convert
        local_goal = global_to_local(self.dest_global_pos, self.global_home)
        local_goal_delta = \
                [local_goal[0] - north_offset,
                local_goal[1] - east_offset]
        grid_goal = tuple([int(np.ceil(x)) for x in local_goal_delta])

        # Run A* to find a path from start to goal
        # DONE: add diagonal motions with a cost of sqrt(2) to your A* implementation
        
        print('Local Start and Goal - ned delta: ', local_start_delta, local_goal_delta)
        print('Local Start and Goal - grid     : ', grid_start, grid_goal)
        
        # DONE: move to a different search space such as a graph (not done here)
        epsilon = 10
        if self.mode == 'grid':
            path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        elif self.mode == 'probabilistic':
            path = self.probabilistic_path
        elif self.mode == 'medial_axis':
            path, _ = optimal_medial_axis(grid, 
                                    heuristic,
                                    grid_start,
                                    grid_goal)
        else:
            path    = test_path(grid_start)

        # Convert path to waypoints
        if self.mode != 'probabilistic':
            waypoints = [[  p[0] + north_offset, 
                            p[1] + east_offset, 
                            TARGET_ALTITUDE, 
                            0] for p in path]
        else:
            waypoints = [[  int(p[0]) , 
                            int(p[1]) , 
                            int(TARGET_ALTITUDE), 
                            0] for p in path]

        # Set heading of wp2 based on relative position to wp1
        for i in range(len(waypoints)-2):
            wp1 = waypoints[i+1][:2]
            wp2 = waypoints[i+2][:2]
            waypoints[i+2][3] = np.arctan2((wp2[1]-wp1[1]), (wp2[0]-wp1[0]))

        # DONE: prune path to minimize number of waypoints
        # DONE (if you're feeling ambitious): Try a different approach altogether!
        print('Prunning waypoints ...')
        waypoints = colinear_reduction(waypoints,epsilon)

        # Set self.waypoints
        self.waypoints = waypoints

        # DONE: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()



if __name__ == "__main__":

    # select one fo act as default destination
    #temp=[-122.397058,37.792669,10] #short distance
    temp=[-122.396905,37.795125,10] #medium distance
    #temp=[-122.393672,37.797160,10] #large distance

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    parser.add_argument('--dest_global_pos_lon', type=float, default=temp[0], help="Destination lon")
    parser.add_argument('--dest_global_pos_lat', type=float, default=temp[1], help="Destination lat")
    parser.add_argument('--dest_global_pos_alt', type=float, default=temp[2], help="Destination alt")
    parser.add_argument('--mode', type=str, default='probabilistic', help="Motion planning: test, grid, probabilistic, medial_axis")
    args = parser.parse_args()

    if args.mode == 'probabilistic':
        # for probabilistic path optimization
        # calculate this path before establishing the MavLink connection
        # to avoid timeout errors
        print("Compute probabilistic waypoints ...")

        with open('colliders.csv') as f_:
            lat0, lon0 = [float(s.split(' ')[1]) for s in str(f_.readline()).split(', ')]
        print(lon0,lat0)

        local_position_ned = global_to_local(np.array([-122.397450,37.792480,0]), np.array([lon0,lat0,0]))
        local_goal = global_to_local(   np.array([ \
                                            args.dest_global_pos_lon,
                                            args.dest_global_pos_lat,
                                            args.dest_global_pos_alt,
                                        ]), 
                                        np.array([lon0,lat0,0]))

        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        probabilistic_path, _ = optimal_probabilistic( data,
                                                    args.dest_global_pos_alt, 
                                                    heuristic, 
                                                    local_position_ned, 
                                                    local_goal)
    else:
        probabilistic_path = None

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)

    drone = MotionPlanning( conn,
                            np.array([  args.dest_global_pos_lon, 
                                        args.dest_global_pos_lat, 
                                        args.dest_global_pos_alt]),
                            args.mode,
                            probabilistic_path)
    time.sleep(1)

    drone.start()
