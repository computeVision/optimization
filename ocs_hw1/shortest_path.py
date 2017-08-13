import sys
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy
import scipy.optimize

def read_input(argv):
    """
    read input based on commandline parameters
    """
    if len(sys.argv) == 1:   # no parameters - load standard files
        fname_v = 'V.txt'
        fname_e = 'E.txt'
        fname_w = 'w.txt'
    elif len(sys.argv) == 4:   # use filenames from the specified parameters
        fname_v = sys.argv[1]
        fname_e = sys.argv[2]
        fname_w = sys.argv[3]
    else:
        print 'Usage: '+sys.argv[0]+' [name_v] [name_e] [name_w]\n name_v, name_e and name_w are optional filenames for vertex, connectivity matrix and weight files respectively'
        sys.exit()
    
    # sanitize input
    V = np.loadtxt(fname_v)
    if V.shape[0] != 2:
        raise RuntimeError('Expected 2xN vertex matrix')
    E = np.loadtxt(fname_e)
    num_vertices = V.shape[1]
    if E.shape != (num_vertices, num_vertices):
        raise RuntimeError('Connectivity matrix is '+str(E.shape)+' should be square according to number of vertices ('+str(num_vertices)+')!')
    num_edges = np.sum(E)
    w = np.loadtxt(fname_w)
    if w.size != num_edges:
        raise RuntimeError('Wrong number of weights (is '+str(w.size)+' should be '+str(num_edges)+')')
        
    return V, E, w

def draw_graph(V, E, w=None, filename=None):
    """
    Draw a graph specified by vertices V and connectivity matrix E.
    w is an optional vector of weights. If specified, weight values are drawn for each edge.
    If filename is specified, write the resulting figure to a file
    """
    plt.figure()
    for idx,p in enumerate(V.T):
        plt.plot(p[0], p[1], 'bo')
        plt.text(p[0], p[1], 'V'+str(idx))
        
    for idx, (e_row, e_col) in enumerate(zip(*np.where(E))):
        plt.plot([V[0,e_row], V[0,e_col]], [V[1,e_row], V[1,e_col]], 'k--')
        if w is not None:
            pos = V[:,e_row] + (V[:,e_col]-V[:,e_row])/2   # midpoint between vertices
            plt.text(pos[0], pos[1], str(round(w[idx],2)), color='b')   # print edge weight
    
    if filename is not None:
        plt.savefig(filename)

if __name__ =='__main__':
    plt.close('all')
    
    V, E, w = read_input(sys.argv)
    num_vertices = V.shape[1]
    num_edges = np.sum(E)

    print "num_edges", num_edges
    # w = -w  # longest way
    #w = np.ones(num_edges)          # uniform weights
    #w = np.random.rand(num_edges)*10  # random weights
    
    draw_graph(V,E,w)

    print "V", V
    print "E", E
    print "w", w

    # TODO: Assemble & solve LP to find shortest path
    a = np.zeros((num_vertices, int(num_edges)))

    edge = 0
    save_e = list()
    for i in xrange(E.shape[0]):
        for j in xrange(E.shape[1]):
            if(E[i][j] >= 1):
                a[i, edge] = 1
                a[j, edge] = -1
                save_e.append((i,j,edge))
                edge = edge + 1

    E_solution = np.zeros_like(E)    # TODO: reconstruct solution connectivity matrix

    b = np.zeros((num_vertices,))
    b[0] = 1
    b[num_vertices - 1] = -1
    x = scipy.optimize.linprog(w,a, b).x

    for i,xi in enumerate(x):
        if(xi >= 1):
            E_solution[save_e[i][0],save_e[i][1]] = 1

    draw_graph(V,E_solution,filename="shortest_path.png")
    plt.show()
