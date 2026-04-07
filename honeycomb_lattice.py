import numpy as np
import numpy.linalg as LA
import matplotlib.path as mpath
import matplotlib.patches as mpatches
Path = mpath.Path

# ---------------------------------------------------------------------------
# High-symmetry k-points for the honeycomb Brillouin zone (lattice constant a=1)
# ---------------------------------------------------------------------------
GAMMA  = np.array([0, 0])
K      = 2 * np.pi * np.array([2/3, 0]) / np.sqrt(3)           # K  point
K_PRIME = 2 * np.pi * np.array([1/3, 1/np.sqrt(3)]) / np.sqrt(3)  # K' point
M_POINT = 2 * np.pi * np.array([1/2, 1/(2*np.sqrt(3))]) / np.sqrt(3)  # M point


# ---------------------------------------------------------------------------
# k-path utilities for band-structure plots
# ---------------------------------------------------------------------------

def get_kvectors(pt1, pt2, num=101):
    """
    Return an array of *num* k-vectors linearly interpolated from *pt1* to *pt2*.

    Parameters
    ----------
    pt1, pt2 : array_like  start and end points in reciprocal space
    num      : int         number of points (default 101)

    Returns
    -------
    k : ndarray, shape (num, 2)
    """
    kx = np.linspace(pt1[0], pt2[0], num=num)
    ky = np.linspace(pt1[1], pt2[1], num=num)
    return np.vstack((kx, ky)).T


def get_path(k):
    """
    Return cumulative arc-length along the k-path *k*.

    Parameters
    ----------
    k : ndarray, shape (N, 2)

    Returns
    -------
    path : ndarray, shape (N,)  cumulative distance from the first point
    """
    dot = np.vectorize(np.dot, signature='(n),(m)->()')
    return np.sqrt(dot(k, k))


def group_kvectors(*segments):
    """
    Concatenate k-path *segments*, removing duplicate endpoints.

    Parameters
    ----------
    *segments : ndarrays  each of shape (N_i, 2)

    Returns
    -------
    k_vectors : ndarray, shape (total_points, 2)
    """
    k_vectors = segments[0]
    for seg in segments[1:]:
        k_vectors = np.concatenate((k_vectors, seg[1:]))
    return k_vectors


def get_total_path(*segments):
    """
    Compute the cumulative arc-length along a multi-segment k-path and
    return the indices of the high-symmetry junctions.

    Parameters
    ----------
    *segments : ndarrays  arc-length arrays for each segment (from get_path)

    Returns
    -------
    path    : ndarray  concatenated arc-length array
    k_index : ndarray  indices of high-symmetry points (including endpoints)
    """
    lengths = np.array([len(s) for s in segments], dtype=float)
    k_index = np.zeros(len(segments) + 1)
    path = np.concatenate(segments)
    k_index[-1] = lengths.sum()
    for i in range(len(segments) - 1):
        if i != 0:
            k_index[i + 1] = k_index[i] + lengths[i]
        else:
            k_index[i + 1] = k_index[i] + lengths[i] - 1
    return path, k_index

def get_reciprocal_vectors(a:np.ndarray, d=2):
    """
    Compute the reciprocal lattice vectors.

     Parameters:
        a (np.ndarray): The lattice vectors.
        d (int): The dimension of the lattice.

    Returns:
        B (np.ndarray): The reciprocal lattice vectors.
    """
    if d == 2:
        A = np.array([a[0],a[1]]).T
    else:
        A = np.array([a[0],a[1],a[2]]).T
    B = 2*np.pi*LA.inv(A)
    return B

def bzmesh(n=200,m=2):
    '''
    Create a meshgrid of k-points in the first Brillouin zone.

     Parameters:
        n (int): Number of points in each direction.
        m (int): Number of times the BZ is repeated in each direction.

    Returns:
        kx (np.ndarray): The x-coordinates of the meshgrid.
        ky (np.ndarray): The y-coordinates of the meshgrid.
    '''
    x = np.linspace(-0.5*m*np.pi,0.5*m*np.pi,2*n+1)
    y = np.linspace(-0.5*m*np.pi,0.5*m*np.pi,2*n+1)
    kx,ky = np.meshgrid(x,y)
    return kx,ky

# need to change later, not a standard way to get honeycom lattice BZ
def honeycomb_bz():
    """
     Parameters:
        None

    Returns:
        honeycomb_bz_x (np.ndarray): The x-coordinates of the honeycomb BZ.
        honeycomb_bz_y (np.ndarray): The y-coordinates of the honeycomb BZ.
    """
    honeycomb_bz_x = 2/3*2*np.pi*np.array([-1/np.sqrt(3),-0.5/np.sqrt(3),0.5/np.sqrt(3),1/np.sqrt(3),0.5/np.sqrt(3),-0.5/np.sqrt(3),-1/np.sqrt(3)])
    honeycomb_bz_y = 2/3*2*np.pi*np.array([0,1/2,1/2,0,-1/2,-1/2,0])
    return honeycomb_bz_x,honeycomb_bz_y

def get_symmetry_pts_index_honeycomb(m=2):
    """
    """
    # symmetry points
    honeycomb_bz_x, honeycomb_bz_y = honeycomb_bz()
    # non repeated symmetry points
    honeycomb_bz_x = np.unique(honeycomb_bz_x)
    honeycomb_bz_y = np.unique(honeycomb_bz_y)
     
    # spacing
    kx,ky = bzmesh(m=2)
    
    index_x = np.zeros(honeycomb_bz_x.shape[0])
    index_y = np.zeros(honeycomb_bz_y.shape[0])
    
    for i in range(honeycomb_bz_x.shape[0]):
        index_x[i] = np.abs(kx[0]-honeycomb_bz_x[i]).argmin()
    for j in range(honeycomb_bz_y.shape[0]):
        index_y[j] = np.abs(ky.T[0]-honeycomb_bz_y[j]).argmin() 
    # make sure the index array is sorted
    index_x.sort()
    index_y.sort()
    
    # from lowest to highest
    K_points = np.array([[index_y[0],index_x[1]],[index_y[0],index_x[-2]],[index_y[1],index_x[0]],
                         [index_y[1],index_x[-1]],[index_y[-1],index_x[1]],[index_y[-1],index_x[-2]]])
    return K_points

def points_in_bz():
    # get symmetry points
    K_points = get_symmetry_pts_index_honeycomb(AFM=AFM)

    # draw path
    verts = np.array([K_points[0],K_points[1],K_points[3],K_points[5],K_points[4],K_points[2],K_points[0]])
    
    codes = [
        Path.MOVETO, # pick up the pen and move to ( , ); related to 1 vertices
        Path.LINETO, # draw the line to from previous ( , ) to current ( , )
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY, # draw a line segment to the start point of current polyline
]
    path = Path(verts, codes, closed=True)
    
def rotation2D(point:np.ndarray, theta:float):
    '''
    Rotate a point in 2D space.

    Parameters:
        point (np.ndarray): The point to rotate.
        theta (float): The angle of rotation.

    Returns:
        R@point (np.ndarray): The rotated point.
    '''
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return R@point

def bz_integration_honeycomb(f_matrix,n=200,m=1):
    '''
    Integrate a function over the first Brillouin zone of a honeycomb lattice.
    
    Parameters:
        f_matrix (np.ndarray): The values of function on lattice to integrate.
        n (int): Number of points in each direction
        m (int): Number of times the BZ is repeated in each direction

    Returns:
        np.sum(f_matrix_in_bz)*dk*dk (float): The integral of the function over the BZ.
    '''
    # spacing
    kx, ky = np.meshgrid(np.linspace(-m*np.pi, m*np.pi, 2*n+1),np.linspace(-m*np.pi, m*np.pi, 2*n+1)) # m = 1 -> 1st; 2nd; etc
    dk = np.abs(kx[0,1]-kx[0,0])

    # BZ of honeycomb
    a = 1 # lattice 
    # found by A^T*B=2pi I, where cols of A and B are the primitive vectors in position and reciprocal space respectively
    M = np.array([0, 1/3])
    K = np.array([2/(3*np.sqrt(3)),0])
    theta = np.pi/3
    
    verts = 2*np.pi/a*np.array([
        K, # right, middle (K)
        rotation2D(K, theta), # right, bottom (K')
        rotation2D(K, 2*theta), # left, bottom
        rotation2D(K, 3*theta), # left, middle
        rotation2D(K, 4*theta), # left, top
        rotation2D(K, 5*theta), # right, top
        K, # closed
    ])
    
    codes = [
        Path.MOVETO, # start
        Path.LINETO, # join
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY, # close
    ]
    
    path = Path(verts, codes, closed=True)

    # filter of points within bz including boundary
    k_points = np.vstack((kx.flatten(), ky.flatten())).T
    grid = path.contains_points(k_points, radius=0) # something different from square
    mask = grid.reshape(2*n+1,2*n+1)

    f_matrix_in_bz = f_matrix[mask]
    f_matrix_in_bz[np.isnan(f_matrix_in_bz)] = 0
    return np.sum(f_matrix_in_bz)*dk*dk

def bz_product_honeycomb(f_matrix,n=200,m=1):
    '''
    Integrate a function over the first Brillouin zone of a honeycomb lattice.
    
    Parameters:
        f_matrix (np.ndarray): The values of function on lattice to integrate.
        n (int): Number of points in each direction
        m (int): Number of times the BZ is repeated in each direction

    Returns:
        np.sum(f_matrix_in_bz)*dk*dk (float): The integral of the function over the BZ.
    '''
    # spacing
    kx, ky = np.meshgrid(np.linspace(-m*np.pi, m*np.pi, 2*n+1),np.linspace(-m*np.pi, m*np.pi, 2*n+1)) # m = 1 -> 1st; 2nd; etc
    dk = np.abs(kx[0,1]-kx[0,0])

    # BZ of honeycomb
    a = 1 # lattice 
    # found by A^T*B=2pi I, where cols of A and B are the primitive vectors in position and reciprocal space respectively
    M = np.array([0, 1/3])
    K = np.array([2/(3*np.sqrt(3)),0])
    theta = np.pi/3
    
    verts = 2*np.pi/a*np.array([
        K, # right, middle (K)w
        rotation2D(K, theta), # right, bottom (K')
        rotation2D(K, 2*theta), # left, bottom
        rotation2D(K, 3*theta), # left, middle
        rotation2D(K, 4*theta), # left, top
        rotation2D(K, 5*theta), # right, top
        K, # closed
    ])
    
    codes = [
        Path.MOVETO, # start
        Path.LINETO, # join
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY, # close
    ]
    
    path = Path(verts, codes, closed=True) # idea is to connect the corner of BZ via the path

    # filter of points within bz including boundary
    k_points = np.vstack((kx.flatten(), ky.flatten())).T
    grid = path.contains_points(k_points, radius=0) # something different from square
    mask = grid.reshape(2*n+1,2*n+1)

    f_matrix_in_bz = f_matrix[mask]
    f_matrix_in_bz[np.isnan(f_matrix_in_bz)] = 0
    return np.prod(f_matrix_in_bz)