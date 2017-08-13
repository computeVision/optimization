import sys
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import graph, solver

def rgb2gray(I):
    return np.dot(I[:,:,0:3], [0.299, 0.587, 0.114])

def load_image(name):
    I = imread(name)
    if len(I.shape)>2:    # convert to grayscale
        I = rgb2gray(I)
    
    if np.max(I) > 1:
        I[np.where(I<128)] = 0    # threshold to get binary image
        I[np.where(I>=128)] = 1
    return I

if __name__ =='__main__':
    
    plt.close('all')
    
    if len(sys.argv) == 2:
        I = load_image(sys.argv[1])
    else:
        I = load_image('labyrinth1.png')
        # I = load_image('labyrinth2.jpg')
        # I = load_image('labyrinth3.jpg')
        # I = load_image('labyrinth4.png')
        # I = load_image('labyrinth5.png')
    
    #plt.figure(); plt.imshow(I,interpolation='none', cmap='gray')
    
    A,b,w = graph.build_graph(I)
    #plt.figure(400)

    # t0 = timer()
    # x_standardLP = solver.solve_standardLP(A.todense(), b, w)
    # graph.visualize_solution(I, x_standardLP)
    # print 'optimization took standard LP', timer()-t0

    t0 = timer()
    print 'start affine solver'
    x_affine, energy = solver.solve_affine_scaling(A, b, w)
    print 'optimization took', timer()-t0
    print "x_affine shape{}".format(x_affine.shape)
    solver.plot_energy(energy)
    plt.savefig('report/energy_plot3.png')
    graph.visualize_solution(I, x_affine, filename='report/shortest_path.png')
    
    plt.show()

