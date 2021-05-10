#! /usr/bin/env python

# imports of external packages to use in our code
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as opt

#setting matplotlib ticksize
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

#set matplotlib global font size
matplotlib.rcParams['font.size']=14


# global variables

# image size
cmin = -50
cmax = 50

# calculates the square distance between
# points and cluster centers
def square_distance(data, centers):
    
    points = np.transpose(np.vstack( [data[:, 0], data[:, 1]] ))

    
    differences = [points - center for center in centers]
    
    sq_differences = np.array( [np.sum(difference**2, axis=1) for difference in differences] )
    
    return sq_differences

# assigns the points to cluster centers by
# square distance
def generate_assignment(square_distances):
    
    correspondence = np.stack(square_distances, axis=1)
    
    assignments = np.argmin(correspondence, axis=1)
    
    return assignments

# calculates the cluster centroids by calculating
# the flux-weighted average of the points
def calculate_centroids(data, assignments, no_centers=2):
    
    points = np.transpose(np.vstack( [data[:, 0], data[:, 1]] ))
    
    weights = data[:, 2]
    
    centroids = []
    
    for i in range(no_centers):
        point_split = points[np.equal(assignments, i)]
        weights_split = weights[np.equal(assignments, i)]
        
        centx = np.sum( weights_split * point_split[:, 0]) / np.sum(weights)
        centy = np.sum( weights_split * point_split[:, 1]) / np.sum(weights)
        
        centroids.append(np.array([centx, centy]))
        
    return np.array(centroids)

# plots the correspondence between cluster centers and points
# plot last every time, plot all if verbose
def plot_correspondence(points, assignments, centers, name='default_name'):
    
    plt.figure(figsize=[12,12])
        
    plt.scatter(points[:, 0], points[:, 1], c=plt.cm.Set1(assignments))
    
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=100,\
                c=plt.cm.Set1(np.arange(0, len(centers))), edgecolors='k')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.title(name)
    
    plt.show()

def k_clustering(data, no_clusters=2, verbose=False, need_plot=False):
    
    # generate random initial center positions
    centers = np.stack([np.random.uniform(low=cmin, high=cmax, size=no_clusters),\
                        np.random.uniform(low=cmin, high=cmax, size=no_clusters)], axis=1)
    # keep track of iterations
    i = 0

    while True:

        square_distances = square_distance(data, centers)

        assignments = generate_assignment(square_distances)
        
        if verbose:
            
            plot_correspondence(data, assignments, centers, name='Iteration {}'.format(i))

        new_centers = calculate_centroids(data, assignments)
        
        # clustering converges when the centers don't change
        if np.array_equal(centers, new_centers):
            
            if need_plot:
                plot_correspondence(data, assignments, centers, name='Iteration {}'.format(i))

            if verbose:
                print('Assignments Converged!')
                print('Centers are:')
                print(centers)
            
            return centers, assignments

        else:

            i += 1
            centers = new_centers

# given k-means assignments and data
# plots the points corresponding to the stars
def plot_stars(image, assignments):
    

    # remove zero-count pixels from data
    counts = image[:, 2]

    data = image[ counts != 0 ]
    
    fig, ax = plt.subplots(2, 3, figsize=[20,15])
    
    star1 = data[assignments == 0]
    star2 = data[assignments == 1]
    
    xlabel = 'X [px]'
    ylabel = 'Y [px]'

    ax[0,1].scatter( image[:, 0], image[:, 1], c=image[:, 2], cmap='gray')
    ax[0,1].set_xlabel(xlabel)
    ax[0,1].set_ylabel(ylabel)
    ax[0,1].set_title('Stars on image')
    ax[1,0].scatter( star1[:, 0], star1[:, 1], c=star1[:, 2], cmap='gray')
    ax[1,0].set_xlabel(xlabel)
    ax[1,0].set_ylabel(ylabel)
    ax[1,0].set_title('Points associated to Star 1')
    ax[1,2].scatter( star2[:, 0], star2[:, 1], c=star2[:, 2], cmap='gray')
    ax[1,2].set_xlabel(xlabel)
    ax[1,2].set_ylabel(ylabel)
    ax[1,2].set_title('Points associated to Star 2')
    ax[0,0].set_axis_off()
    ax[0,2].set_axis_off()
    ax[1,1].set_axis_off()
    

    plt.show()

# masks data by filtering out data points that
# are below a given threshold
def threshold_mask(data, threshold=0.01):
    
    counts = data[:, 2]
    
    count_max = np.amax(counts)
    
    masked = data[ counts > threshold * count_max ]
    
    return masked


# fitting a general 2D Gaussian to our data
def gauss2d(xy, amp, x0, y0, a, b, c):
    x = xy[0]
    y = xy[1]
    arg = a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2
    
    return amp * np.exp(-arg)

# does the function fitting
def fit_gauss2d(data):
    
    max_arg = np.argmax( data[:, 2] )
    
    guess = [1, data[max_arg, 0], data[max_arg, 1], 1, 1, 1]
    
    x = data[:, 0]
    y = data[:, 1]
    xy = [x, y]
    
    pred_params, uncert_cov = opt.curve_fit( gauss2d, xy, data[:, 2], p0=guess)
    
    return pred_params, uncert_cov

# gets the center position from Gaussian fitting
def get_predicted_mean(data):
    
    pred, cov = fit_gauss2d(data)
    
    unc = np.sqrt(np.diag(cov))
    
    x = pred[1]
    y = pred[2]
    
    ux = unc[1]
    uy = unc[2]
    
    return np.array( [x, y] ), np.array( [ux, uy] )

# variance weighted average
def weighted_avg(x, u):
    
    w = 1/u**2
    
    xb = np.sum(x * w) / np.sum(w)
    
    err = np.sqrt(1/np.sum(w))
    
    return xb, err

# calculates the separation distance on the image
def calculate_separation_distance(s1, s2, u1, u2):
    
    x1 = s1[:, 0]
    y1 = s1[:, 1]
    
    x2 = s2[:, 0]
    y2 = s2[:, 1]
    
    dx1 = u1[:, 0]
    dy1 = u1[:, 1]
    
    dx2 = u2[:, 0]
    dy2 = u2[:, 1]
    
    sep = np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
    
    # this formula comes from formal error propagation
    dsep = np.sqrt(\
                  ((dx1**2 + dx2**2)*(x1-x2)**2 + (dy1**2+dy2**2)*(y1-y2)**2)/\
                  ((x1-x2)**2 + (y1-y2)**2))
    
    return sep, dsep

# main function for our Python code
if __name__ == "__main__":
   
    haveInput = False
    InputFile = None
    verbose = False
    threshold = 0.001
    
    # reading in the cmd args
    for i in range(1,len(sys.argv)):
        if sys.argv[i] == '-h' or sys.argv[i] == '--help':
            continue

        # seeing if we have input files
        if sys.argv[i] == '--input':
            InputFile = sys.argv[i+1]
            haveInput = True

        # verbose output?
        if sys.argv[i] == '--verbose':
            verbose = True
        
        # custom threshold
        if '--threshold' in sys.argv:
            p = sys.argv.index('--threshold')
            ptemp = float(sys.argv[p+1])
            if ptemp > 0 and ptemp < 1:
                threshold = ptemp

    if '-h' in sys.argv or '--help' in sys.argv or not np.all(haveInput):
        print ("Usage: %s [options] --input [input data]" % sys.argv[0])
        print ("  options:")
        print ("   --help(-h)          print options")
        print ("   --threshold         [number between 0 and 1] masking threshold")
        print ("   --verbose           displays many plots")
        sys.exit(1)
    
    # reading in data from files
    images = np.load(InputFile)

    Nexp = images.shape[0]
    Nmeas = images.shape[1]
    

    star1_pos = []
    star1_unc = []
    star2_pos = []
    star2_unc = []
    
    need_plot = True

    # loop over all experiments
    for n in range(Nexp):
        
        pos1 = []
        pos2 = []
        unc1 = []
        unc2 = []
        
        # loop over all images
        for image in images[n]:

            # remove zero-count pixels from data
            counts = image[:, 2]

            data = image[ counts != 0 ]

            centers, assignments = k_clustering(data, verbose=verbose, need_plot=need_plot)

            if verbose or need_plot:
                plot_stars( image, assignments ) 
                need_plot = False

            # stars are associated with each cluster of points

            star1 = data[ assignments == 0 ]
            star2 = data[ assignments == 1 ]

            # filter out pixels with too few counts

            signal1 = threshold_mask( star1 , threshold=threshold)
            signal2 = threshold_mask( star2 , threshold=threshold)

            # fit the Gaussian to find predicted center and uncertainty

            m1, u1 = get_predicted_mean(signal1)

            m2, u2 = get_predicted_mean(signal2)

            # we aren't guaranteed to get the same star assigned to
            # star1 or star2
            # fill in the first time, then do a comparison to decide
            # which one is which

            if len(pos1) == 0:

                pos1.append(m1)
                pos2.append(m2)
                unc1.append(u1)
                unc2.append(u2)

            else:

                # positions to compare with
                ref1 = pos1[0]
                ref2 = pos2[0]

                #only need to calculate square distance for one point
                d1 = np.square( m1 - ref1 ).sum()

                d2 = np.square( m1 - ref2 ).sum()

                # if ref1 is closer to m1, then
                # we have the same star in ref1 and m1
                if d1 < d2:

                    pos1.append(m1)
                    pos2.append(m2)
                    unc1.append(u1)
                    unc2.append(u2)

                # otherwise, we need to swap them
                else:
                    pos1.append(m2)
                    pos2.append(m1)
                    unc1.append(u2)
                    unc2.append(u1)

        # combine all measurements using the variance weighted average

        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        unc1 = np.array(unc1)
        unc2 = np.array(unc2)

        xbar1, xerr1 = weighted_avg(pos1[:, 0], unc1[:, 0])
        ybar1, yerr1 = weighted_avg(pos1[:, 1], unc1[:, 1])

        xbar2, xerr2 = weighted_avg(pos2[:, 0], unc2[:, 0])
        ybar2, yerr2 = weighted_avg(pos2[:, 1], unc2[:, 1])

        position1 = np.array( [xbar1, ybar1] )
        uncertainty1 = np.array( [xerr1, yerr1] )

        position2 = np.array( [xbar2, ybar2] )
        uncertainty2 = np.array( [xerr2, yerr2] )
        
        # we aren't guaranteed to get the same star assigned to
        # star1 or star2
        # fill in the first time, then do a comparison to decide
        # which one is which

        if len(star1_pos) == 0:

            star1_pos.append(position1)
            star2_pos.append(position2)
            star1_unc.append(uncertainty1)
            star2_unc.append(uncertainty2)

        else:

            # positions to compare with
            ref1 = star1_pos[0]
            ref2 = star2_pos[0]

            #only need to calculate square distance for one point
            d1 = np.square( position1 - ref1 ).sum()

            d2 = np.square( position1 - ref2 ).sum()

            # if ref1 is closer to m1, then
            # we have the same star in ref1 and m1
            if d1 < d2:

                star1_pos.append(position1)
                star2_pos.append(position2)
                star1_unc.append(uncertainty1)
                star2_unc.append(uncertainty2)

            # otherwise, we need to swap them
            else:
                star1_pos.append(position2)
                star2_pos.append(position1)
                star1_unc.append(uncertainty2)
                star2_unc.append(uncertainty1)

    star1_pos = np.array(star1_pos)
    star1_unc = np.array(star1_unc)
    star2_pos = np.array(star2_pos)
    star2_unc = np.array(star2_unc)
    
    # plot the star positions from each experiment
    
    plt.figure(figsize=[10,10])

    plt.errorbar(star1_pos[:, 0], star1_pos[:, 1], xerr=star1_unc[:, 0], yerr=star1_unc[:, 1], fmt='bo')
    plt.errorbar(star2_pos[:, 0], star2_pos[:, 1], xerr=star2_unc[:, 0], yerr=star2_unc[:, 1], fmt='ro')
    plt.title('Predicted position for each experiment')
    plt.xlabel('x [px]')
    plt.ylabel('y [px]')
    plt.show()
    
    # calculate the separation distance
    
    sep, dsep = calculate_separation_distance(star1_pos, star2_pos, star1_unc, star2_unc)
    
    print('Separation distances are:')
    print(sep)
    print('Errors on each are:')
    print(dsep)
    
    np.savetxt('separation_distances.txt', [sep, dsep])
    print('Saved result to separation_distances.txt')

    sys.exit(1)

