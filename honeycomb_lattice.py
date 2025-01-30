import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
Path = mpath.Path

def bzmesh(n=200,m=2):
    '''
    Create BZ meshgrid
    m = 2 for standard BZ of square lattice [-pi,pi]x[-pi,pi] =(2pi)^2  
    '''
    x = np.linspace(-0.5*m*np.pi,0.5*m*np.pi,2*n+1)
    y = np.linspace(-0.5*m*np.pi,0.5*m*np.pi,2*n+1)
    kx,ky = np.meshgrid(x,y)
    return kx,ky

# need to change later, not a standard way to get honeycom lattice BZ
def honeycomb_bz():
    """
        return corners of the honeycomb lattice BZ
    """
    honeycomb_bz_x = 2/3*2*np.pi*np.array([-1/np.sqrt(3),-0.5/np.sqrt(3),0.5/np.sqrt(3),1/np.sqrt(3),0.5/np.sqrt(3),-0.5/np.sqrt(3),-1/np.sqrt(3)])
    honeycomb_bz_y = 2/3*2*np.pi*np.array([0,1/2,1/2,0,-1/2,-1/2,0])
    return honeycomb_bz_x,honeycomb_bz_y

def get_symmetry_pts_index_honeycomb(m=2):
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
    
def rotation2D(point, theta):
    '''
    anticlockwise rotation with angle = theta, in radian
    '''
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return R@point

def bz_integration_honeycomb(f_matrix,n=200,m=1):
    '''
    surface integration on honeycomb BZ
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