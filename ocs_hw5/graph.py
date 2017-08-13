
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def get_exits(I):
    """
    Find exits in the labyrinth image. Only labyrinths with exits on top/bottom
    are supported. I is an image where value 0 means blocked, values different from 0
    mean free.
    
    Parameters:
    -----------
    
    I : M x N array
    
    Returns:
    --------
    
    E : 2 x 2 array
        pixel coordinates (y,x) of exits are in the rows of E
    """
    E = []
    top = np.where(I[0,:])
    for x in top[0]:
        E.append([0,x])
    
    bottom = np.where(I[-1,:])
    for x in bottom[0]:
        E.append([I.shape[0]-1,x])
    
    return np.array([E[0],E[-1]])
    
def sub2ind(I, s):
    """
    Subscripts to linear index. Computes for an index (i,j) (corresponds to pixel
    coordinate (j,i)) the corresponding linear index in the graph, i.e. index (i,j)
    corresponds to the k-th graph node.
    Only works for free pixels (I[i,j] != 0) !!
    """
    free_y,free_x = np.where(I)
    candidates = np.where(free_y==s[0])[0]   # search for specified y coordinate among 
                                            # all free pixels
    if candidates.size == 0:
        raise RuntimeError('Couldn\'t find any free pixels with y='+str(s[0]))
    offset = candidates[0]                   # remember offset
    match = np.where(free_x[candidates] == s[1])[0]
    if match.size == 0:
        raise RuntimeError('Couldn\'t find any free pixels with y='+str(s[0])+' and x='+str(s[1]))
    ind = offset + match[0]   # search for x coordinate
    
    return ind

def subs2inds(I):
    """
    Make a dictionary that maps indices of free pixels to the linear index of the
    corresponding node in the graph. Since arrays are not hashable, use the string
    representation of the coordinates list as key.
    
    Example:
    --------
    Obtain the index of the pixel at coordinate (36,15) (i.e. index (15,36) - the 
    pixel must be free!
    
    >>> s2i = subs2inds(I)
    >>> p = [15,36]
    >>> idx = s2i[str(p)]
    """
    free_y, free_x = np.where(I)
    subs2inds = {}
    for idx,(x,y) in enumerate(zip(free_x,free_y)):
        p = [y,x]
        subs2inds[str(p)] = idx
        
    return subs2inds

    
def build_graph(I):
    """
    Build graph, constraint matrix A, vector b and weight vector w given by the 
    labyrinth image. Only free pixels are considered in the graph.
    Edges are given by a 4-neighborhood, ordered clockwise starting
    with the edge to the pixel above.
    Pixel to node mapping is row-wise, i.e. similar to a C-style flattening.
    
    Parameters:
    -----------
    
    I : M x N array
        image
    
    Returns:
    --------
    
    tuple (A,b,w) : Constraint matrix A, vector b and weights w
    """
    M,N = I.shape
    free_y,free_x = np.where(I)  # y/x indices of free pixels
    start, end = get_exits(I)    
    
    num_vertices = free_x.size   # as many vertices in the graph as free pixels
    
    s2i = subs2inds(I)
    i = []; j = []; v = []   # row indices, column indices and values for sparse matrix
    print 'Building graph...'

    for idx,(y,x) in enumerate(zip(free_y,free_x)):
        if free_y.size>200:
            if idx%(free_y.size/100*10) == 0:
                print '%.2f done' % (float(idx)/free_y.size*100)

        if y>0:
            if I[y-1,x] == I[y,x]:    # pixel above is free
                i.append(idx); j.append(idx*4); v.append(1)          # outgoing edge to pixel above
                i.append(idx); j.append(s2i[str([y-1,x])]*4 + 2); v.append(-1)      # incoming edge from pixel above
        
        if x<N-1:
            if I[y,x+1] == I[y,x]:   # pixel to the right is free
                i.append(idx); j.append(idx*4 + 1); v.append(1)   # outgoing edge to right pixel
                i.append(idx); j.append((idx+1)*4 + 3); v.append(-1)  # incoming edge from right pixel

        if y<M-1:
            if I[y+1,x] == I[y,x]:   # pixel below is free
                i.append(idx); j.append(idx*4 + 2); v.append(1)   # outgoing edge to pixel below
                i.append(idx); j.append(s2i[str([y+1,x])]*4); v.append(-1)  # incoming edge from pixel below
    
        if x>0:
            if I[y,x-1] == I[y,x]:    # pixel to the left is free
                i.append(idx); j.append(idx*4 + 3); v.append(1)   # outgoing edge to left pixel
                i.append(idx); j.append((idx-1)*4 + 1); v.append(-1)   # incoming edge from left pixel

                
    b = np.zeros(num_vertices)     # as many constraints as vertices
    b[sub2ind(I,start)] = 1
    b[sub2ind(I,end)] = -1

    w = np.ones(num_vertices*4)     # as many weights as edges
    w += np.random.rand(w.size)*0.1   # make solution unique
    
    A = coo_matrix((v, (i, j)), shape=(num_vertices, num_vertices*4)).tocsr()

    return A,b,w
    
def visualize_solution(I, x, filename=None):
    """
    Visualize a path in the image I given by the solution vector x
    """
    plt.figure(); plt.imshow(I, interpolation='none', cmap='gray')

    free_y,free_x = np.where(I)
    edges = np.reshape(x,(-1,4))    # every row in the matrix corresponds to a graph node
    
    th = 0.5
    e_row, e_col = np.where(edges > th)   # find all rows with active edges
    
    for ne, (idx, e) in enumerate(zip(e_row, e_col)):
        x = free_x[idx]; y = free_y[idx]   # x/y coordinates of node
        if e==0:
            plt.plot([x,x],[y,y-1],'b', lw=2.5)
        elif e==1:
            plt.plot([x,x+1],[y,y],'b', lw=2.5)
        elif e==2:
            plt.plot([x,x],[y,y+1],'b', lw=2.5)
        elif e==3:
            plt.plot([x,x-1],[y,y],'b', lw=2.5)
        
        if ne % 250 == 0:
            plt.pause(0.001)
        
    plt.pause(0.001)
    if filename is not None:
        plt.savefig(filename)
