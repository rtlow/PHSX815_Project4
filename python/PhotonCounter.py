#! /usr/bin/env python

# imports of external packages to use in our code
import sys
import numpy as np
import scipy.special as special

# global variables

# images we produce will be 100x100
# so coordinates will run from [-50,50]

cmin = -50
cmax = 50

# definite star positions [pixel]
star1_x = 25
star1_y = 30

star2_x = -15
star2_y = -5

mean_seeing = 3 # standard deviation of atmospheric seeing [pixel]

I1 = 10000 # maximum intensity of star 1 [counts/second]
I2 = 7000 # maximum intensity of star 2 [counts/second]

# generate all possible coordinate pairs
def generate_coords(x, y):

    coords = np.empty( (len(x), len(y), 2), dtype=np.intp )
    coords[..., 0] = x[:, None]
    coords[..., 1] = y
    
    # need to reshape this output to get a 2D array
    return coords.reshape( len(x)**2, 2 )

# 2D Airy Disk
# x: array of [x,y] ordered pairs
# x0: array of [x0, y0] star center coordinates
# I: maximum intensity of source
def AiryDisk(x, x0, I):
    
    xc = x[:, 0]
    yc = x[:, 1]
    
    q = np.sqrt( (xc - x0[0])**2 + (yc - x0[1])**2 )

    result = I * (2 * special.j1(q) / q ) ** 2

    # we will get a divide by zero when the center pixel is inputted
    # we know that that maximum value is just the maximum intensity, I0
    intensities = np.nan_to_num( result, nan=I )

    return intensities.reshape( (len(intensities), 1) )

# producing samples from the Airy Disk

def GenerateImage(Nmeas):

    # need the +1 to include the endpoint
    x = np.arange(cmin, cmax + 1)
    y = np.arange(cmin, cmax + 1)

    coords = generate_coords(x, y)

    data = []

    for n in range(Nmeas):

        # generate a random star position for each star
        # centered at their centers, with mean seeing

        # approximate seeing as Gaussian
        x1 = np.random.normal(star1_x, mean_seeing)
        y1 = np.random.normal(star1_y, mean_seeing)

        x2 = np.random.normal(star2_x, mean_seeing)
        y2 = np.random.normal(star2_y, mean_seeing)
        
        # generate the intensity for each star independently
        # the Airy Disk intensity there will translate into
        # a Poisson rate parameter, which we will use to 
        # generate a random intensity
        # the total image is just the superposition
        i1 = np.random.poisson( AiryDisk( coords, [x1, y1], I1 ) )
        i2 = np.random.poisson( AiryDisk( coords, [x2, y2], I2 ) )

        intensities = i1 + i2
        
        # package the coordinates with the intensities
        image_data = np.hstack( [coords, intensities] )

        data.append(image_data)

    return data


# main function for experiment code
if __name__ == "__main__":
    # if the user includes the flag -h or --help print the options
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s [options]" % sys.argv[0] )
        print('Options:')
        print('-Ii [number]         max intensity of ith (0, 1) star in counts/second')
        print('-seeing [number]     mean seeing of atmosphere in arcseconds')
        print('-Nmeas [number]      no. measurements per experiment')
        print('-Nexp [number]       no. experiments')
        print('-output [string]     output file name')
        print
        sys.exit(1)



    # default number of exposures (letting light collect for fixed time) - per experiment
    Nmeas = 1

    # default number of experiments
    Nexp = 1

    # output file defaults
    doOutputFile = False



    if '-Nmeas' in sys.argv:
        p = sys.argv.index('-Nmeas')
        Nt = int(sys.argv[p+1])
        if Nt > 0:
            Nmeas = Nt
    if '-Nexp' in sys.argv:
        p = sys.argv.index('-Nexp')
        Ne = int(sys.argv[p+1])
        if Ne > 0:
            Nexp = Ne
    if '-I1' in sys.argv:
        p = sys.argv.index('-I1')
        Ne = int(sys.argv[p+1])
        if Ne > 0:
            I1 = Ne
    if '-I2' in sys.argv:
        p = sys.argv.index('-I2')
        Ne = int(sys.argv[p+1])
        if Ne > 0:
            I2 = Ne
    if '-seeing' in sys.argv:
        p = sys.argv.index('-I0')
        Ne = int(sys.argv[p+1])
        if Ne > 0:
            MeanSeeing = Ne
    if '-output' in sys.argv:
        p = sys.argv.index('-output')
        OutputFileName = sys.argv[p+1]
        doOutputFile = True

    experiments = []


    for n in range(Nexp):

        measurements = GenerateImage(Nmeas)
        experiments.append(measurements)

    if doOutputFile:

        np.save(OutputFileName, experiments)

    else:
        print( experiments )
