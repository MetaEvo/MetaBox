"""
=========================================================
File Name: uav_numpy.py
Author: Sijie Ma (GitHub: Feather_red)
Date: 2025-02-17
=========================================================

Description:
This script implements UAV path planning based on the paper:
"Benchmarking Global Optimization Techniques for Unmanned Aerial Vehicle Path Planning."
It provides a Python migration of a MATLAB implementation.

Features:
- Implements UAV path planning in a real-world scenario.
- Uses NumPy for numerical computations.
- Translates MATLAB code to Python.

Dependencies:
- numpy
- torch
- pickle

References:
- Shehadeh, M. A., & Kudela, J. (2025). Benchmarking global optimization techniques for unmanned aerial vehicle path planning.
  arXiv. https://arxiv.org/abs/2501.14503

Version:
- Python implementation using NumPy

=========================================================
"""
from ....problem.basic_problem import Basic_Problem
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pickle

class UAV_Numpy_Problem(Basic_Problem):
    def __init__(self):
        self.terrain_model = None
        self.FES = 0
        self.optimum = None
        self.problem_id = None
        self.dim = None
        self.lb = None
        self.ub = None

    def __str__(self):
        return f"Terrain {self.problem_id}"

    def __boundaries__(self):
        model = self.terrain_model

        nVar = model['n']

        # Initialize the boundaries dictionaries
        VarMin = {'x': model['xmin'], 'y': model['ymin'], 'z': model['zmin'],
                  'r': 0, 'psi': -np.pi / 4, 'phi': None}
        VarMax = {'x': model['xmax'], 'y': model['ymax'], 'z': model['zmax'],
                  'r': None, 'psi': np.pi / 4, 'phi': None}

        # Calculate the radial distance range based on the model's start and end points
        distance = np.linalg.norm(np.array(model['start']) - np.array(model['end']))
        VarMax['r'] = 2 * distance / nVar

        # Inclination (elevation) limits (angle range is pi/4)
        AngleRange = np.pi / 4
        VarMin['psi'] = -AngleRange
        VarMax['psi'] = AngleRange

        # Azimuth (phi)
        dirVector = np.array(model['end']) - np.array(model['start'])
        phi0 = np.arctan2(dirVector[1], dirVector[0])
        VarMin['phi'] = (phi0 - AngleRange).item()
        VarMax['phi'] = (phi0 + AngleRange).item()

        # Lower and upper Bounds of velocity
        alpha = 0.5
        VelMax = {'r': alpha * (VarMax['r'] - VarMin['r']),
                  'psi': alpha * (VarMax['psi'] - VarMin['psi']),
                  'phi': alpha * (VarMax['phi'] - VarMin['phi'])}
        VelMin = {'r': -VelMax['r'],
                  'psi': -VelMax['psi'],
                  'phi': -VelMax['phi']}

        # Create bounds by stacking both position and velocity limits
        bounds = np.array([
            [VarMin['r'], VarMax['r']],
            [VarMin['psi'], VarMax['psi']],
            [VarMin['phi'], VarMax['phi']],
        ])
        # Since we are interested in r, phi, psi for nVar points in each item of the population
        bounds = np.tile(bounds, (int(nVar), 1))

        # Assign the 0th column to self.lb and the 1st column to self.ub
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]

    def spherical_to_cart_vec(self, solve):
        # solve : 2D [NP, 3 * n]
        # Extracting r, phi, and psi from the solution row
        r = solve[:, 0::3]  # r values are in indices 0, 3, 6, ...
        psi = solve[:, 1::3]  # psi values are in indices 1, 4, 7, ...
        phi = solve[:, 2::3]  # phi values are in indices 2, 5, 8, ...
        # [NP, n]
        model = self.terrain_model
        xs, ys, zs = model['start'] # 1D

        # [NP, n]
        x = np.zeros_like(r)
        y = np.zeros_like(r)
        z = np.zeros_like(r)

        x[:, 0] = xs + r[:, 0] * np.cos(psi[:, 0]) * np.sin(phi[:, 0])
        y[:, 0] = ys + r[:, 0] * np.cos(psi[:, 0]) * np.cos(phi[:, 0])
        z[:, 0] = zs + r[:, 0] * np.sin(psi[:, 0])

        x[:, 0] = np.clip(x[:, 0], model['xmin'], model['xmax'])
        y[:, 0] = np.clip(y[:, 0], model['ymin'], model['ymax'])
        z[:, 0] = np.clip(z[:, 0], model['zmin'], model['zmax'])

        # Next Cartesian coordinates
        for i in range(1, x.shape[1]):
            x[:, i] = x[:, i - 1] + r[:, i] * np.cos(psi[:, i]) * np.sin(phi[:, i])
            x[:, i] = np.clip(x[:, i], model['xmin'], model['xmax'])

            y[:, i] = y[:, i - 1] + r[:, i] * np.cos(psi[:, i]) * np.cos(phi[:, i])
            y[:, i] = np.clip(y[:, i], model['ymin'], model['ymax'])

            z[:, i] = z[:, i - 1] + r[:, i] * np.sin(psi[:, i])
            z[:, i] = np.clip(z[:, i], model['zmin'], model['zmax'])

        return x, y, z

    def DistP2S(self, xs, a, b):
        # xs: 1D array [2], a: 2D array [2, NP], b: 2D array [2, NP]

        # Convert x to a 2D array of shape [2, 1] to match the other input arrays
        x = xs[:, None]  # Shape: [2, 1]

        # Compute the Euclidean distances between points a and b, a and x, b and x
        d_ab = np.linalg.norm(a - b, axis = 0)  # Distance between points a and b: [NP]
        d_ax = np.linalg.norm(a - x, axis = 0)  # Distance between points a and x: [NP]
        d_bx = np.linalg.norm(b - x, axis = 0)  # Distance between points b and x: [NP]

        # Initialize the distance: If d_ab == 0, use d_ax, otherwise use the minimum of d_ax and d_bx
        dist = np.where(d_ab == 0, d_ax, np.minimum(d_ax, d_bx))

        # Compute the dot product between vectors (b - a) and (x - a)
        dot_product_ab_ax = np.sum((b - a) * (x - a), axis = 0)  # Dot product: [NP]

        # Compute the dot product between vectors (a - b) and (x - b)
        dot_product_ba_bx = np.sum((a - b) * (x - b), axis = 0)  # Dot product: [NP]

        # Calculate the perpendicular distance from point x to the line segment ab
        dist = np.where(
            (d_ab != 0) & (dot_product_ab_ax >= 0) & (dot_product_ba_bx >= 0),
            np.abs(np.cross((b - a), (x - a), axis = 0)) / d_ab,  # Perpendicular distance
            dist  # If not within the valid region, retain previous distance
        )

        return dist

class Terrain(UAV_Numpy_Problem):
    def __init__(self, terrain_model, problem_id):
        super(Terrain, self).__init__()
        self.terrain_model = terrain_model
        self.dim = 3 * terrain_model['n']
        self.problem_id = problem_id
        self.optimum = None
        self.__boundaries__()
        # self.SphCost = eng.CreateSphCost(self.terrain_model)


    def func(self, solve):
        # solve : 2D [NP, 3 * nv]
        model = self.terrain_model

        J_pen = model['J_pen']
        H = model['H']

        x, y, z = self.spherical_to_cart_vec(solve) # 2D : [NP, nv]

        NP, n = x.shape
        # Start location
        xs, ys, zs = model['start'] # 1D

        # Final location
        xf, yf, zf = model['end'] # 1D

        # REPEAT
        xs = np.tile(xs, (NP, 1))
        ys = np.tile(ys, (NP, 1))
        zs = np.tile(zs, (NP, 1))

        xf = np.tile(xf, (NP, 1))
        yf = np.tile(yf, (NP, 1))
        zf = np.tile(zf, (NP, 1))

        # Concatenate
        x_all = np.concatenate((xs, x, xf), axis = 1)
        y_all = np.concatenate((ys, y, yf), axis = 1)
        z_all = np.concatenate((zs, z, zf), axis = 1)

        N = x_all.shape[1] # Full path length

        # Altitude wrt sea level = z_relative + ground_level
        z_abs = np.zeros((NP, N))
        for i in range(N):
            x_index = np.round(x_all[:, i]).astype(int) - 1
            y_index = np.round(y_all[:, i]).astype(int) - 1
            z_abs[:, i] = z_all[:, i] + H[y_index, x_index]

        # ---------- J1 Cost for path length ----------
        diff = np.stack((x_all[:, 1:] - x_all[:, :-1], y_all[:, 1:] - y_all[:, :-1], z_abs[:, 1:] - z_abs[:, :-1]), axis = -1)  # [NP, N-1, 3]
        J1 = np.sum(np.linalg.norm(diff, axis = 2), axis = 1)

        # ---------- J2 - threats / obstacles Cost ----------
        threats = model['threats']
        threat_num = threats.shape[0]

        # Checking if UAV passes through a threat
        J2 = np.zeros(NP)
        for i in range(threat_num):
            threat = threats[i, :]
            threat_x = threat[0]
            threat_y = threat[1]
            threat_radius = threat[3]

            for j in range(N - 1):
                dist = self.DistP2S(np.array([threat_x, threat_y]), np.array([x_all[:, j], y_all[:, j]]), np.array([x_all[:, j + 1], y_all[:, j + 1]]))
                # dist 1D NP
                threat_cost = threat_radius + model['drone_size'] + model['danger_dist'] - dist # Dangerous Zone
                threat_cost = np.where(dist > threat_radius + model['drone_size'] + model['danger_dist'], 0, threat_cost) # No Collision
                threat_cost = np.where(dist < threat_radius + model['drone_size'], J_pen, threat_cost) # Collision

                J2 += threat_cost

        # ---------- J3 - Altitude cost ----------
        z_max = model['zmax']
        z_min = model['zmin']
        J3 = np.sum(np.where(z < 0, J_pen, np.abs(z - (z_max + z_min) / 2)), axis = 1)

        # ---------- J4 - Smooth cost ----------
        J4 = np.zeros(NP)
        turning_max = 45
        climb_max = 45

        # Calculate the projections of the segments in the xy-plane (x, y, 0) for all paths at once
        diff_1 = np.stack((x_all[:, 1:] - x_all[:, :-1], y_all[:, 1:] - y_all[:, :-1], np.zeros((NP, N - 1))), axis = -1)  # [NP, N-1, 3]
        diff_2 = np.stack((x_all[:, 2:] - x_all[:, 1:-1], y_all[:, 2:] - y_all[:, 1:-1], np.zeros((NP, N - 2))), axis = -1)  # [NP, N-2, 3]

        for i in range(0, N - 2):
            segment1_proj = diff_1[:, i, :] # [NP, 3]
            segment2_proj = diff_2[:, i, :] # [NP, 3]

            # Find rows where all values in segment1_proj are zero (no movement in this segment)
            zero_segment1 = np.all(segment1_proj == 0, axis = 1)  # [NP,] - boolean array, True where all are zeros

            # Find rows where all values in segment2_proj are zero (no movement in this segment)
            zero_segment2 = np.all(segment2_proj == 0, axis = 1)  # [NP,] - boolean array, True where all are zeros

            # Handle zero segments: if a segment is all zeros, use the previous or next valid segment
            # For segment1_proj, if it's zero, we will use the previous segment (diff_1[i-1])
            i1 = i - 1
            i2 = i + 1
            while i1 >= 0 and np.any(zero_segment1):
                segment1_proj[zero_segment1] = diff_1[zero_segment1, i1, :]
                zero_segment1 = np.all(segment1_proj == 0, axis = 1)
                i1 -= 1

            while i2 < N - 2 and np.any(zero_segment2):
                segment2_proj[zero_segment2] = diff_2[zero_segment2, i2, :]
                zero_segment2 = np.all(segment2_proj == 0, axis = 1)
                i2 += 1

            # segment1_proj and segment2_proj [NP, 3]
            # Calculate the climb angles
            climb_angle1 = np.degrees(np.arctan2(z_abs[:, i + 1] - z_abs[:, i], np.linalg.norm(segment1_proj, axis = 1)))
            climb_angle2 = np.degrees(np.arctan2(z_abs[:, i + 2] - z_abs[:, i + 1], np.linalg.norm(segment2_proj, axis = 1)))

            # Calculate the turning angle
            turning_angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(segment1_proj, segment2_proj, axis = 1), axis = 1),
                                                  np.sum(segment1_proj * segment2_proj, axis = 1)))

            addition_J_1 = np.where(np.abs(turning_angle) > turning_max, np.abs(turning_angle), 0)
            addition_J_2 = np.where(np.abs(climb_angle2 - climb_angle1) > climb_max, np.abs(climb_angle2 - climb_angle1), 0)

            J4 += addition_J_1 + addition_J_2

        # ---------- J5 - terrain cost ----------
        J5 = np.full(NP, J_pen)
        paths_above = self.are_paths_clear(x_all, y_all, z_abs, H)
        J5[paths_above] = 0
        b1 = 5
        b2 = 1
        b3 = 10
        b4 = 1
        b5 = 1
        return np.array([b1 * J1, b2 * J2, b3 * J3, b4 * J4, b5 * J5]).T

    def are_paths_clear(self, x_all, y_all, z_abs, H, num_samples = 12):
        """
        Check if all the line segments connecting adjacent points of NP paths are completely above the terrain H.
        :param x_all: (NP, N) shaped array, x coordinates of N points for each of NP paths
        :param y_all: (NP, N) shaped array, y coordinates of N points for each of NP paths
        :param z_abs: (NP, N) shaped array, absolute heights of N points for each of NP paths
        :param H: (H_rows, H_cols) shaped terrain height matrix
        :param num_samples: The number of sample points along each line segment (excluding the endpoints)
        :return: A boolean array of shape (NP,) where each value indicates whether all segments of a path are above the terrain
        """
        # Generate the interpolation function for the terrain

        H_rows, H_cols = H.shape
        x_indices = np.arange(H_cols)  # X-direction indices
        y_indices = np.arange(H_rows)  # Y-direction indices
        interp_func = RegularGridInterpolator((y_indices, x_indices), H, bounds_error = False, fill_value = np.nan)

        # Calculate the sample points for all line segments
        x_interp = np.linspace(x_all[:, :-1], x_all[:, 1:], num_samples, axis = 2)  # (NP, N-1, num_samples)
        y_interp = np.linspace(y_all[:, :-1], y_all[:, 1:], num_samples, axis = 2)
        z_interp = np.linspace(z_abs[:, :-1], z_abs[:, 1:], num_samples, axis = 2)

        # Compute the terrain heights at all interpolated points
        terrain_heights = interp_func(np.stack([y_interp, x_interp], axis = -1))  # (NP, N-1, num_samples)

        # Check if all sampled points are above the terrain
        return np.all(z_interp > terrain_heights, axis = (1, 2))

if __name__ == "__main__":
    x =  [
        [
            102.787067274924, -0.588464384157333, 0.523312419417429, 8.98171798143932,
            0.0707168743046490, 1.49658000170548, 80.6110279168768, -0.195561226893039,
            0.574004754481592, 141.203633522607, -0.369189314002375, 0.551542337270883,
            179.697141403396, 0.604469599761478, 0.521125312856793, 112.889900716559,
            -0.153310866363274, 0.599568876578121, 156.479010612955, 0.515890617157642,
            0.761018950795744, 158.169662571115, -0.159932706752021, 0.242214520089881,
            94.8344807895242, -0.650786093007384, 1.46528689191431, 20.9142143722353,
            -0.173403956357546, 0.724803312220412
        ],
        [
            102.787067274924, -0.588464384157333, 0.523312419417429, 8.98171798143932,
            0.0707168743046490, 1.49658000170548, 80.6110279168768, -0.195561226893039,
            0.574004754481592, 141.203633522607, -0.369189314002375, 0.551542337270883,
            179.697141403396, 0.604469599761478, 0.521125312856793, 112.889900716559,
            -0.153310866363274, 0.599568876578121, 156.479010612955, 0.515890617157642,
            0.761018950795744, 158.169662571115, -0.159932706752021, 0.242214520089881,
            94.8344807895242, -0.650786093007384, 1.46528689191431, 20.9142143722353,
            -0.173403956357546, 0.724803312220412
        ]
    ]
    pkl_file = "UAE_terrain_data/Model56.pkl"
    with open(pkl_file, 'rb') as f:
        model_data = pickle.load(f)
    terrain_data = model_data[4]
    terrain_data['n'] = 10
    terrain_data['J_pen'] = 1e4

    F5 = Terrain(terrain_data, 5)
    np.set_printoptions(10)
    x_np = np.array(x)
    cost = F5.func(x_np)
    print(cost)

    # add weight
    Cost = 5 * cost[:, 0] + 1 * cost[:, 1] + 10 * cost[:, 2] + 1 * cost[:, 3]
    print(Cost)