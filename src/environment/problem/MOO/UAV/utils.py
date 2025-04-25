import numpy as np
from scipy.interpolate import LinearNDInterpolator

def createmodel(map_size = 900, r = 100, rr = 10, num_threats = 5, rng = None):
    """
        Generates a simulation model for a drone navigation task in a terrain with threats.

        This function creates a synthetic terrain map and places cylindrical threats randomly.
        It also determines suitable start and end points for the drone, ensuring they are not
        in immediate collision with any threats.

        Parameters:
        ----------
        map_size : int, optional (default=900)
            Size of the square terrain grid.
        r : float, optional (default=100)
            Initial roughness of the terrain.
        rr : float, optional (default=10)
            Roughness variation factor controlling terrain detail.
        num_threats : int, optional (default=5)
            Number of cylindrical threats to be placed on the map.
        seed : int, optional (default=3849)
            Random seed for reproducibility.

        Returns:
        -------
        model : dict
            A dictionary containing terrain, threat information, and start/end points:

            - 'drone_size' : float
              The size of the drone.
            - 'danger_dist' : float
              Minimum safety distance from threats.
            - 'threats' : ndarray, shape (num_threats, 4)
              Array where each row represents a threat: `[x, y, height, radius]`.
            - 'start' : ndarray, shape (3, 1)
              Start position `[x, y, z]` for the drone.
            - 'end' : ndarray, shape (3, 1)
              End position `[x, y, z]` for the drone.
            - 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax' : float
              Boundaries of the map.
            - 'MAPSIZE_X', 'MAPSIZE_Y' : float
              Dimensions of the terrain grid.
            - 'X', 'Y' : ndarray
              Meshgrid representing terrain coordinates.
            - 'H' : ndarray
              Heightmap of the terrain.

        Notes:
        -----
        - The threats are randomly placed with constraints ensuring they remain within bounds.
        - Start and end points are selected to avoid initial collisions with threats.
        - If no valid start/end points are found after 20 attempts, an assertion error is raised.

        Example:
        -------
        ```
        ```
        """
    H = generate_terrain(8, map_size, 130, r, rr, rng)
    MAPSIZE_X = H.shape[1]
    MAPSIZE_Y = H.shape[0]
    X, Y = np.meshgrid(np.arange(1, MAPSIZE_X + 1), np.arange(1, MAPSIZE_Y + 1))

    # The drone's size
    drone_size = 1.0
    model = {'drone_size': drone_size,
             'danger_dist': 10 * drone_size
             }

    # Threats as cylinders
    threats = rng.rand(num_threats, 4)
    threats[:, 3] = (np.round(MAPSIZE_X / 15 + threats[:, 3] * MAPSIZE_X / 30))  # R IN [60, 90]
    threats[:, 0] = (np.round(np.minimum(threats[:, 3], 0.1 * MAPSIZE_X) + threats[:, 0] * 0.7 * MAPSIZE_X))  # With 15% offset from right left-hand
    threats[:, 1] = (np.round(np.minimum(threats[:, 3], 0.1 * MAPSIZE_Y) + threats[:, 1] * 0.9 * MAPSIZE_Y))
    threats[:, 2] = (np.round(H[threats[:, 1].astype(int), threats[:, 0].astype(int)] - 10 + threats[:, 2] * 40))  # Height from -10 to 30 above actual center
    model['threats'] = threats.copy()

    # Map limits
    xmin = 1.0
    xmax = float(MAPSIZE_X)

    ymin = 1.0
    ymax = float(MAPSIZE_Y)

    zmin = 100.0
    zmax = 200.0

    # collision check
    coll_check = np.zeros(num_threats)
    Mission_impossible = 0
    while not np.all(coll_check):
        xs = np.round((0.02 + rng.rand() * 0.17) * MAPSIZE_X)
        xe = np.round((1 - 0.02 - rng.rand() * 0.17) * MAPSIZE_X)

        ys = np.round((0.02 + rng.rand() * 0.17) * MAPSIZE_Y)
        ye = np.round((1 - 0.02 - rng.rand() * 0.17) * MAPSIZE_Y)

        model['start'] = np.array([xs, ys, 150])[:, None]
        model['end'] = np.array([xe, ye, 150])[:, None]
        dist = np.linalg.norm(np.stack((threats[:, 0] - xs, threats[:, 1] - ys), axis = 1), axis = 1)
        coll_check = np.where(dist > threats[:, 3] + drone_size + model['danger_dist'], 1, 0)

        Mission_impossible = Mission_impossible + 1
        if Mission_impossible > 20:
            assert "No possible start/end points to be found."

    model['xmin'] = xmin
    model['xmax'] = xmax
    model['zmin'] = zmin
    model['zmax'] = zmax
    model['ymin'] = ymin
    model['ymax'] = ymax
    model['MAPSIZE_X'] = float(MAPSIZE_X)
    model['MAPSIZE_Y'] = float(MAPSIZE_Y)
    model['X'] = X.astype(float)
    model['Y'] = Y.astype(float)
    model['H'] = H.astype(float)

    return model

def generate_terrain(n = 8, map_size = 900, h0 = 130, r0 = None, rr = None, rng = None):
    """
        Generates a series of points that approximate terrain using a simple algorithm
        with minimal parameters.

        Parameters:
        ----------
        n : int
            Number of iterations of the algorithm, controlling the level of detail.
            Values beyond 8 add negligible visible detail but significantly increase computation time.
        mesh_size : int
            The size of the output mesh (e.g., 512 for a 512x512 grid).
        h0 : float
            Initial elevation.
        r0 : float
            Initial roughness, determining how much terrain can vary in a step.
        rr : float
            Roughness roughness, controlling how much roughness itself can vary in a step.

        Returns:
        -------
        hm : ndarray
            2D mesh grids useful for surface plotting.
        """
    n0 = rng.randint(low = 1, high = 5)
    m = 3
    nf = n0 * (m + 1) ** n

    # Create initial x, y, and height coordinates and roughness map
    x = np.concatenate((rng.randn(n0), np.zeros(nf - n0)))
    y = np.concatenate((rng.randn(n0), np.zeros(nf - n0)))
    h = np.concatenate((r0 * rng.randn(n0) + h0, np.zeros(nf - n0)))
    r = np.concatenate((rr * rng.randn(n0) + r0, np.zeros(nf - n0)))

    # Create new points from old points n times
    for k in range(1, n + 1):
        # Calculate the new variance for the x, y random draws and for the h, r random draws.
        dxy = 0.75 ** k
        dh = 0.5 ** k

        # Number of new points to generate
        n_new = m * n0

        # Parents for new points
        parents = np.tile(np.arange(n0), (m, 1)).flatten('F')

        # Calculate indices for new and existing points.
        new = np.arange(n0, n0 + n_new)
        old = np.arange(n0)

        # Generate new x y values.
        theta = 2 * np.pi * rng.rand(n_new)
        radius = dxy * (rng.rand(n_new) + 1)
        x[new] = x[parents] + radius * np.cos(theta)
        y[new] = y[parents] + radius * np.sin(theta)
        # todo index
        # Interpolate to find nominal new r and h values and add noise to roughness and height maps.
        r[new] = interpolate(x[old], y[old], r[old], x[new], y[new]) + (dh * rr) * rng.randn(n_new)
        h[new] = interpolate(x[old], y[old], h[old], x[new], y[new]) + (dh / dxy) * radius * r[new] * rng.randn(n_new)

        n0 = n_new + n0

    x = (x - np.median(x)) / np.std(x)
    y = (y - np.median(y)) / np.std(y)

    xm, ym = np.meshgrid(np.linspace(-1, 1, map_size), np.linspace(-1, 1, map_size))
    hm = interpolate(x, y, h, xm.flatten(), ym.flatten()).reshape(map_size, map_size)
    return hm

def interpolate(x0, y0, v0, xn, yn):

    x_ext = np.concatenate([100 * np.array([-1, -1, 1, 1]), x0.flatten()])
    y_ext = np.concatenate([100 * np.array([-1, 1, -1, 1]), y0.flatten()])
    v_ext = np.concatenate([np.zeros(4), v0.flatten()])

    interp = LinearNDInterpolator(list(zip(x_ext, y_ext)), v_ext)
    return interp(xn, yn)