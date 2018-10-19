from enum import Enum
from queue import PriorityQueue
import numpy as np
import utm
import time
import numpy as np

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

import networkx as nx
print('nx.__version__',nx.__version__) # should be 2.1
from shapely.geometry import Polygon, Point, LineString
import numpy.linalg as LA
from sklearn.neighbors import KDTree
from skimage.morphology import medial_axis
from skimage.util import invert
from scipy.spatial import Voronoi, voronoi_plot_2d

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

def test_path(grid_start):
    path = [[grid_start[0]+i,grid_start[1]] for i in range(5)]
    path += [[grid_start[0]+4,grid_start[1]+1+i] for i in range(5)]
    path += [[grid_start[0]+5+i,grid_start[1]+5] for i in range(5)]
    return path

def colinear_reduction(wp,epsilon):
    if len(wp) < 3:
        return wp
    elif len(wp) == 3:
        if collinearity_float(wp[0],wp[1],wp[2],epsilon):
            return [wp[0],wp[2]]
        else:
            return wp
    else:
        if collinearity_float(wp[0],wp[1],wp[2],epsilon):
            return colinear_reduction([wp[0]]+wp[2:],epsilon)
        else:
            return [wp[0]]+colinear_reduction(wp[1:],epsilon)

def point(p):
    return np.array([p[0], p[1], 1.])

def collinearity_float(p1, p2, p3, epsilon=1e-6): 
    collinear = False
    mat = np.vstack((point(p1), point(p2), point(p3)))
    det = np.linalg.det(mat)
    if abs(det) < epsilon:
        collinear = True
    return collinear

def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """
    NORTH =  (-1, 0, 1)
    NE =     (-1, 1, np.sqrt(2))
    EAST =   ( 0, 1, 1)
    SE =     ( 1, 1, np.sqrt(2))
    SOUTH =  ( 1, 0, 1)
    SW =     ( 1,-1, np.sqrt(2))
    WEST =   ( 0,-1, 1)
    NW =     (-1,-1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    if x - 1 < 0 or y + 1 > m or grid[x-1, y + 1] == 1:
        valid_actions.remove(Action.NE)
    if x + 1 > n or y + 1 > m or grid[x+1, y + 1] == 1:
        valid_actions.remove(Action.SE)
    if x + 1 > n or y - 1 < 0 or grid[x+1, y - 1] == 1:
        valid_actions.remove(Action.SW)
    if x - 1 < 0 or y - 1 < 0 or grid[x-1, y - 1] == 1:
        valid_actions.remove(Action.NW)

    return valid_actions


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        #plt.plot(current_node[1], current_node[0], '.')
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

def a_star_graph(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""
    
    # TODO: complete

    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    queue.put((new_cost, next_node))
                    
                    branch[next_node] = (new_cost, current_node)
             
    path = []
    path_cost = 0
    if found:
        
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
            
    return path[::-1], path_cost

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def sampler(data, TARGET_ALTITUDE):

    def extract_polygons(data):
        polygons = []
        for i in range(data.shape[0]):
            north, east, alt, d_north, d_east, d_alt = data[i, :]
            obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
            corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]), (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]
            height = alt + d_alt
            p = Polygon(corners)
            polygons.append((p, height))
        return polygons

    print('Extracting polygons ...')
    polygons = extract_polygons(data)
    print(len(polygons))

    print('Calculating centroids ...')
    centroids = []
    for (p, height) in polygons:
        centroids.append([p.centroid.x,p.centroid.y])
    print(len(centroids))

    print('Sampling ...')
    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])

    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    num_samples = 500

    xvals = np.random.uniform(xmin, xmax, num_samples)
    yvals = np.random.uniform(ymin, ymax, num_samples)
    zvals = np.random.uniform(TARGET_ALTITUDE, TARGET_ALTITUDE, num_samples)

    samples = np.array(list(zip(xvals, yvals, zvals)))

    # ## Removing Points Colliding With Obstacles

    def collides(polygons, point, centroids):  
        idx = -1
        for (p, height) in polygons:
            idx += 1
            centroid = centroids[idx]
#            if Point(point[:2]).distance(p.centroid) < 50:
            if abs(point[0]-centroid[0]) + abs(point[1]-centroid[1]) < 50:
                if p.contains(Point(point)) and height >= point[2]:
                    return True
        return False

    def collides_copy(polygons, point):   
        for (p, height) in polygons:
            if p.contains(Point(point)) and height >= point[2]:
                return True
        return False

    # Use `collides` for all points in the sample.
    print('Removing collisions ...')    
    to_keep = []
    for point in samples:
        #print(point)
        if not collides(polygons, point, centroids):
            #to_keep.append(point)
            to_keep.append(tuple(point))

    return to_keep, polygons, centroids

def optimal_probabilistic(data, TARGET_ALTITUDE, heuristic, start, goal):

    print('Sampling the zone ...')
    nodes, polygons, centroids = sampler(data, TARGET_ALTITUDE)

    def can_connect(n1, n2, centroids):
        l = LineString([n1, n2])
        #for p in polygons:
        #    if p.crosses(l) and p.height >= min(n1[2], n2[2]):
        idx = -1
        for (p, height) in polygons:
            idx += 1
            centroid = centroids[idx]
#            if Point(point[:2]).distance(p.centroid) < 50:
            if  (abs(n1[0]-centroid[0]) + abs(n1[1]-centroid[1]) < 300) or \
                (abs(n2[0]-centroid[0]) + abs(n2[1]-centroid[1]) < 300):
                if p.crosses(l) and height >= min(n1[2], n2[2]):
                    return False
        return True
    def create_graph(nodes, k, centroids):
        g = nx.Graph()
        tree = KDTree(nodes)
        for n1 in nodes:
            # for each node connect try to connect to k nearest nodes
            idxs = tree.query([n1], k, return_distance=False)[0]
            
            for idx in idxs:
                n2 = nodes[idx]
                if n2 == n1:
                    continue
                    
                if can_connect(n1, n2, centroids):
                    g.add_edge(n1, n2, weight=1)
        return g

    print('Creating graph ...')
    t0 = time.time()
    g = create_graph(nodes, 10, centroids)
    print('graph took {0} seconds to build'.format(time.time()-t0))
    print("Number of edges", len(g.edges))

    print('Creating start/goal nodes ...')
    tree = KDTree(nodes)              
    idx_start = tree.query([start], k=1, return_distance=False)[0][0]
    start = nodes[idx_start]

    idx_goal = tree.query([goal], k=1, return_distance=False)[0][0]
    goal = nodes[idx_goal]

    print('start{0}'.format(start))
    print('goal{0}'.format(goal))

    print('Finding optimal path in graph ...')
    path, cost = a_star_graph(g, heuristic, start, goal)
    print("Path length = {0}, path cost = {1}".format(len(path), cost))

    return path, cost

def optimal_medial_axis(grid,heuristic,start,goal):

    print('Creating Medial Axis ...')    
    skeleton = medial_axis(invert(grid))

    def find_start_goal(skel, start, goal):
        skel_cells = np.transpose(skel.nonzero())
        start_min_dist = np.linalg.norm(np.array(start) - np.array(skel_cells), axis=1).argmin()
        near_start = skel_cells[start_min_dist]
        goal_min_dist = np.linalg.norm(np.array(goal) - np.array(skel_cells), axis=1).argmin()
        near_goal = skel_cells[goal_min_dist]
        
        return near_start, near_goal

    skel_start, skel_goal = find_start_goal(skeleton, start, goal)
    print('start{0}, start_skel{1}'.format(start,skel_start))
    print('goal {0}, goal_skel {1}'.format(goal,skel_goal))

    # Run A* on the skeleton
    print('Finding optimal path in graph ...')
    path, cost = a_star(invert(skeleton).astype(np.int), heuristic, tuple(skel_start), tuple(skel_goal))
    print("Path length = {0}, path cost = {1}".format(len(path), cost))

    """
    plt.imshow(grid, cmap='Greys', origin='lower')
    plt.imshow(skeleton, cmap='Greys', origin='lower', alpha=0.7)
    plt.plot(start[1], start[0], 'x')
    plt.plot(goal[1], goal[0], 'x')
    pp = np.array(path)
    plt.plot(pp[:, 1], pp[:, 0], 'g')

    plt.xlabel('EAST')
    plt.ylabel('NORTH')
    plt.show()
    """

    return path, cost


