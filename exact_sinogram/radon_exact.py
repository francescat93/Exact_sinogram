import numpy as np
import numpy.matlib as npmat
import math


def my_phantomgallery( phantom_type ):
    """
    Calculates the matrix of the elements of the phantom given its type.

    Parameters
    ----------
    phantom_type: 'ellipses' (or 'shepp_logan'),'modified_shepp_logan','squares','rectangles'

    Returns
    -------
    M : matrix of the elements of the phantom
    """

    if phantom_type == 'ellipses' or 'shepp_logan':
        # [semiaxis 1, semiaxis 2, x center, y center, phi=angle, greyscale=attenuation]
        p1 = [.7, .8, 0, 0, 0, 1]
        p2 = [.65,.75,0,0,0,-.9]
        p3 = [.15,.2,0,.4,0,.5]
        p4 = [.25,.15,-.25,.25,2.37,.2]
        p5 = [.25,.15,.25,.25,.79,.2]
        p6 = [.08,.25,0,-.3,.5,.65]
        p7 = [.05,.05,.5,-.3,0,.8]
        # combine into a matrix with one ellipse in each row
        M = np.array([p1, p2, p3, p4, p5, p6, p7]);

    elif phantom_type == 'modified_shepp_logan':
        # [semiaxis 1, semiaxis 2, x center, y center, phi=angle, greyscale=attenuation]
        M =  np.array([[  .69,   .92,    0,      0,   0, 1.],
            [ .6624, .8740,    0, -.0184,   0, -.8],
            [ .1100, .3100,  .22,      0, -18, -.2],
            [ .1600, .4100, -.22,      0,  18, -.2],
            [ .2100, .2500,    0,    .35,   0, .1],
            [ .0460, .0460,    0,     .1,   0, .1],
            [ .0460, .0460,    0,    -.1,   0, .1],
            [ .0460, .0230, -.08,  -.605,   0, .1],
            [ .0230, .0230,    0,  -.606,   0, .1],
            [ .0230, .0460,  .06,  -.605,   0, .1]])

    elif phantom_type == 'squares':
        #if 0 each square is a 5-D vector [x0,y0,w,phi=angle,greyscale=attenuation]
        s1 = [0,0,1.3,0,1]
        s2 = [0,0,1.1,0,-.9]
        s3 = [.1,-.1,.5,np.pi/6,.4]
        s4 = [-.25,.15,.25,np.pi/4,.2]
        s5 = [-.2,.25,.3,np.pi/3,.4]
        #combine into a matrix with one square in each row
        M = np.array([s1, s2, s3, s4, s5]);

    elif (phantom_type == 'rectangles'):
        # [x center, y center, dimension 1, dimension 2, phi=angle, greyscale=attenuation]
        r1 = [0,0,1.3,1.1,0,1]
        r2 = [0,0,1.2,1,0,-.9]
        r3 = [0.25,.15,.25,.6,np.pi/6,.4]
        r4 = [-.2,.1,.25,.20,np.pi/4,.2]
        r5 = [-.3,.2,.3,.2,np.pi/6,.4]
        #combine into a matrix with one square in each row
        M = np.array([r1, r2, r3, r4, r5])
    else:
        M = None

    return M


def phantom_ellipses(n_points,E):
    """
    Function that create the phantom image of 'ellipses' type, from the matrix of the elements and given the number of pixels.

    Parameters
    ----------
    n_points:  number of pixels on each row and column
    E:         matrix of the elements of the phantom

    Returns
    -------
    phantom : phantom image
    """

    x,y = np.meshgrid(np.arange(-1,1,2./n_points),np.arange(-1,1,2./n_points))
    nrow,ncol = E.shape

    phantom1 = np.zeros((y.shape[0], y.shape[1], nrow))

    for k in range(nrow): #itero sulle ellissi
        x_new = x - E[k,2]
        y_new = y - E[k,3]


        cond = (E[k,1]**2)*np.square(x_new * math.cos(E[k,4]) + y_new * math.sin(E[k,4])) + \
               (E[k,0]**2)*np.square(y_new * math.cos(E[k,4]) - x_new * math.sin(E[k,4])) - (E[k,0]*E[k,1])**2

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if (cond[i,j] < 0.0):
                    phantom1[i,j,k] = E[k,5];   # gray scale
                else:
                    phantom1[i,j,k] = 0.0;
    phantom1 = phantom1.sum(axis=2)
    phantom = np.flipud(phantom1)
    return phantom

def phantom_squares(n_points,S):
    """
    Function that create the phantom image of 'squares' type, from the matrix of the elements and given the number of pixels.

    Parameters
    ----------
    n_points:  number of pixels on each row and column
    S:         matrix of the elements of the phantom

    Returns
    -------
    phantom : phantom image
    """
    x,y = np.meshgrid(np.arange(-1,1,2/n_points),np.arange(-1,1,2/n_points))  
    nrow,ncol = S.shape
    phantom1 = np.zeros((y.shape[0], y.shape[1], nrow))                        

    for k in range(nrow): #itero sui quadrati
        x_new = x - S[k,0]
        y_new = y - S[k,1]

        u = abs(x_new*math.cos(S[k,3])+y_new*math.sin(S[k,3]))
        v = abs(-x_new*math.sin(S[k,3])+y_new*math.cos(S[k,3]))

        cond = np.maximum(u,v)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if (cond[i,j] < S[k,2]/2):
                    phantom1[i,j,k] = S[k,4];   # gray scale
                else:
                    phantom1[i,j,k] = 0.0;
                #endif
            #endfor
        #endfor
    #endfor

    phantom1 = phantom1.sum(axis=2)
    phantom = np.flipud(phantom1)
    return phantom

def phantom_rectangles(n_points,R):
    """
    Function that create the phantom image of 'rectangles' type, from the matrix of the elements and given the number of pixels.

    Parameters
    ----------
    n_points:  number of pixels on each row and column
    R:         matrix of the elements of the phantom

    Returns
    -------
    phantom : phantom image
    """
    x,y = np.meshgrid(np.arange(-1,1,2/n_points),np.arange(-1,1,2/n_points))
    nrow,ncol = R.shape
    phantom1 = np.zeros((y.shape[0], y.shape[1], nrow))

    for k in range(nrow):         #itero sui rettangoli
        x_new = x - R[k,0]
        y_new = y - R[k,1]

        u = abs(x_new*math.cos(R[k,4])+y_new*math.sin(R[k,4]))
        v = abs(-x_new*math.sin(R[k,4])+y_new*math.cos(R[k,4]))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if (u[i,j] < R[k,2]/2 and v[i,j] < R[k,3]/2):
                    phantom1[i,j,k] = R[k,5];        # gray scale
                else:
                    phantom1[i,j,k] = 0.0;
                #endif
            #endfor
        #endfor
    #endfor

    phantom1 = phantom1.sum(axis=2)
    phantom = np.flipud(phantom1)
    return phantom

def my_radon_analytic(phantom_type, N, theta_vec, M ,  tvec_set=None, circle=False ):
    """
    Function that returns the analytical_sinogram given the phantom.

    Parameters
    ----------
    phantom_type : type of the phantom ('ellipses', 'shepp_logan', 'modified_shepp_logan','squares'rectangles')
    theta_vec    : list of the angles
    M            : matrix of the structure of the phantom
    tvec_set     : vector of the t values given from by the user
    circle       : as in the function iradon of scikit-image "assume the reconstructed image is zero outside the inscribed circle. Also changes the default output_size to match the behaviour of radon called with circle=True."

    Returns
    -------
    analytical_sinogram : Analytical Sinogram of the given phantom
    """

    if phantom_type in ['ellipses', 'shepp_logan', 'modified_shepp_logan']:
            analytical_sinogram = radon_ellipses(N,theta_vec,M, tvec_set,circle);
    elif phantom_type== 'squares':
            analytical_sinogram = radon_squares(N,theta_vec,M, circle);
    elif phantom_type== 'rectangles':
            analytical_sinogram = radon_rectangles(N,theta_vec,M, circle);
    else:
        print('error on the choice of phantom type')
    #endif
    return analytical_sinogram



def radon_ellipses(N,theta_vec, E, tvec_set=None, circle=False):
    """
    Function that compute the analytical_sinogram for phantoms of ellipses type

    Parameters
    ----------
    N         : dimension of the image
    theta_vec : vector of the angles theta
    E         : matrix of the ellipses parameters
    tvec_set  : vector of the t values given from by the user
    circle    : as in the function iradon of scikit-image "assume the reconstructed image is zero outside the inscribed circle. Also changes the default output_size to match the behaviour of radon called with circle=True."

    Returns
    -------
    analytical_sinogram : Analytical Sinogram
    """


    [t_vec, grid_t, grid_theta] = build_t_theta(N,theta_vec, tvec_set=tvec_set, circle =circle);

    (nrowE,ncolE) = E.shape;
    tmp = np.zeros((nrowE,len(grid_theta)));
    for i in range(nrowE):
        grid_theta_new = grid_theta - E[i,4];
        grid_t_new     = (grid_t - E[i,2]*np.cos(grid_theta)-E[i,3]*np.sin(grid_theta))/E[i,1];

        v1 = np.sin(grid_theta_new)**2+((E[i,0]/E[i,1])**2)*np.cos(grid_theta_new)**2 - grid_t_new**2;
        cond = v1;
        v2 = np.zeros((v1.shape[0],1));
        for j in range (len(grid_theta)):
            if cond[j] > 0:
                v2[j]=1;
            else:
                v2[j]=0;
            #endif
        #endfor
        v3 = np.sqrt(v1*v2);
        v4 = np.sin(grid_theta_new)**2+((E[i,0]/E[i,1])**2)*np.cos(grid_theta_new)**2;
        tmp[i,:] = np.transpose( E[i,0]*E[i,5]*(v3/v4) );
    #endfor
    radvec = np.sum(tmp,axis = 0);
    analytical_sinogram = np.transpose(np.reshape(radvec,(len(theta_vec),len(t_vec))));
    return  analytical_sinogram


def radon_squares(N,theta_vec,S, circle=False):
    """
    Function that compute the analytical_sinogram for phantoms of square type

    Parameters
    ----------
    N         : dimension of the image
    theta_vec : list of the angles
    S         : matrix of the squares parameters
    circle    : as in the function iradon of scikit-image "assume the reconstructed image is zero outside the inscribed circle. Also changes the default output_size to match the behaviour of radon called with circle=True."

    Returns
    -------
    analytical_sinogram : Analytical Sinogram
    """
    [t_vec, grid_t, grid_theta] = build_t_theta(N,theta_vec, circle = circle);
    [nrow,ncol] = np.shape(S);
    tmp = np.zeros((nrow,len(grid_theta)));
    for i in range(nrow):       # cycle on the elements of the phantom
        grid_theta_new = grid_theta - S[i,3];
        grid_t_new     = (grid_t - S[i,0]* np.cos(grid_theta) - S[i,1]*np.sin(grid_theta))*2/S[i,2];

        for j in range(len(grid_theta)): # angles
            theta_new = grid_theta_new[j]
            t_new     = grid_t_new[j]
            if theta_new == 0:
                if abs(t_new)< 1:
                    v1= -1;
                    v2= 1;
                else:
                    v1= 0;
                    v2= 0;
                #endif
            else:
                v1= (t_new*np.cos(theta_new)-1)/np.sin(theta_new);
                v2= (t_new*np.cos(theta_new)+1)/np.sin(theta_new);
            #endif

            if theta_new == np.pi/2:
                if abs(t_new)< 1:
                    h1= -1;
                    h2= 1;
                else:
                    h1= 0;
                    h2= 0;
                #endif
            else:
                h1 = (1-t_new*np.sin(theta_new))/np.cos(theta_new);
                h2 = (-1-t_new*np.sin(theta_new))/np.cos(theta_new);
            #endif
            vmax= np.maximum(v1,v2);   # scalar values
            vmin= np.minimum(v1,v2);
            hmax= np.maximum(h1,h2);
            hmin= np.minimum(h1,h2);
            entryval= np.maximum(vmin,hmin);
            exitval= np.minimum(vmax,hmax);

            if (exitval-entryval) > 0:
                tmp[i,j]=(.5)*S[i,4]*S[i,2]*(exitval-entryval);
            else:
                tmp[i,j]=0;
            #endif
        #endfor
    #endfor
    radvec = np.sum(tmp,axis=0);
    radvec = radvec/np.amax(radvec);

    analytical_sinogram = np.transpose(np.reshape(radvec,(len(theta_vec),len(t_vec))));

    return  analytical_sinogram




def radon_rectangles(N,theta_vec,R, circle = False):
    """
    Function that compute the analytical_sinogram for phantoms of rectangle type

    Parameters
    ----------
    N        : dimension of the image
    theta_vec: list of the angles
    R        : matrix of the rectangle parameters
    circle   : as in the function iradon of scikit-image "assume the reconstructed image is zero outside the inscribed circle. Also changes the default output_size to match the behaviour of radon called with circle=True."

    Returns
    -------
    analytical_sinogram : Analytical Sinogram
    """
    [t_vec, grid_t, grid_theta] = build_t_theta(N,theta_vec, circle=circle);

    (nrow, ncol) = R.shape;
    tmp = np.zeros((nrow,len(grid_theta)));
    for i in range(nrow):
        m = R[i,2]/2;
        n = R[i,3]/2;
        grid_theta_new = grid_theta - R[i,4];
        grid_t_new     = (grid_t - R[i,0]*np.cos(grid_theta)- R[i,1]*np.sin(grid_theta));
        for j in range(len(grid_theta)):
            theta_new = grid_theta_new[j];
            t_new     = grid_t_new[j];
            if theta_new== 0:
                if abs(t_new)< m:
                    v1 = -n;
                    v2 = n;
                else:
                    v1 = 0;
                    v2 = 0;
                #endif
            else:
                v1 = (t_new*np.cos(theta_new)- m)/np.sin(theta_new);
                v2 = (t_new*np.cos(theta_new)+ m)/np.sin(theta_new);
            #endif
            if theta_new == np.pi/2:
                if abs(t_new)< n:
                    h1 = - m;
                    h2 = m;
                else:
                    h1 = 0;
                    h2 = 0;
                #endif
            else:
                h1 = (n - t_new*np.sin(theta_new))/np.cos(theta_new);
                h2 = (- n - t_new*np.sin(theta_new))/np.cos(theta_new);
            #endif
            vmax = np.maximum(v1,v2);
            vmin = np.minimum(v1,v2);
            hmax = np.maximum(h1,h2);
            hmin = np.minimum(h1,h2);
            entryval = np.maximum(vmin,hmin);
            exitval = np.minimum(vmax,hmax);

            if (exitval - entryval) > 0:
                tmp[i,j] = R[i,5]*(exitval-entryval);
            else:
                tmp[i,j] = 0;
            #endif
        #endfor
    #endfor
    radvec = np.sum(tmp,axis = 0);
    radvec = radvec/np.amax(radvec);
    analytical_sinogram = np.transpose(np.reshape(radvec,(len(theta_vec),len(t_vec))));

    return  analytical_sinogram



def build_t_theta(N,theta_vec, tvec_set=None, circle=False):
    """
    Function that compute the grid of t and theta for the radon trasform

    Parameters
    ----------
    N        : dimension of the image
    theta_vec: list of the angles
    tvec_set : vector of the t values given from by the user
    circle   : as in the function iradon of scikit-image "assume the reconstructed image is zero outside the inscribed circle. Also changes the default output_size to match the behaviour of radon called with circle=True."

    Returns
    -------
    t_vec       : vector of the t values
    grid_t      : grid of the t values
    grid_theta  : grid of the theta values
    """

    Nangle = len(theta_vec)

    if tvec_set is None:
        dt = 2./(N-1)
        if not circle:
            N = int(np.ceil(N*np.sqrt(2)))
            t_vec = np.sqrt(2) *np.linspace(-1+dt,1-dt,N)
        else:
            t_vec = np.linspace(-1+dt,1-dt,N)
    else:
        t_vec = tvec_set

    # now make a "t, theta" grid:
    grid_t = npmat.repmat(t_vec,1,len(theta_vec))
    grid_t = np.transpose(grid_t)

    #grid_theta = np.zeros((Nrays,1))
    grid_theta = np.tile(theta_vec,(len(t_vec),1)).T.flatten()
    grid_theta = grid_theta.reshape(-1,1)

    return t_vec, grid_t, grid_theta



class Phantom:
    """ Base class for Phantoms
    """
    def __init__( self, phantom_type = 'ellipses', circle = True, matrix = None):
        """
        Parameters and Attributes
        -------------------------
        phantom_type : string, can be 'ellipses', 'shepp_logan', 'modified_shepp_logan','squares','rectangles'
        circle       : as in the function iradon of scikit-image "assume the reconstructed image is zero outside the inscribed circle. Also changes the default output_size to match the behaviour of radon called with circle=True."
        matrix       : matrix of the phantom
        """
        self.phantom_type = phantom_type
        self.circle = circle
        if matrix is None:
            self.matrix = my_phantomgallery(phantom_type)

    def get_phantom( self, N = 128 ):
        """ Compute the matrix phantom image.

        Parameters
        ----------
        N : number of pixels of the image in each dimension

        Returns
        -------
        P : phantom image
        """
        if self.phantom_type in ['ellipses', 'shepp_logan', 'modified_shepp_logan']:
            P = phantom_ellipses(N, self.matrix)
        elif self.phantom_type == 'squares':
            P = phantom_squares(N, self.matrix)
        elif self.phantom_type == 'rectangles':
            P = phantom_rectangles(N, self.matrix)
        else:
            print('Error in the choice of the phantom')
        return P

    def get_sinogram( self, N = 128, theta_vec = None ):
        """ Return the Analytical Sinogram of the phantom.

        Parameters
        ----------
        N         : number of pixels of the image in each dimension
        theta_vec : vector of the angles theta

        Returns
        -------
        analytical_sinogram : matrix of the Analyitical Sinogram of the phantom
        """

        if theta_vec is None:
            theta_vec = np.deg2rad(np.linspace(0, 359, 360))
        if self.phantom_type in ['ellipses', 'shepp_logan', 'modified_shepp_logan']:
                analytical_sinogram = radon_ellipses(N, theta_vec, self.matrix, circle=self.circle);
        elif self.phantom_type == 'squares':
                analytical_sinogram = radon_squares(N, theta_vec, self.matrix, circle = self.circle);
        elif self.phantom_type == 'rectangles':
                analytical_sinogram = radon_rectangles(N, theta_vec, self.matrix,  circle=self.circle);
        else:
            print('error on the choice of phantom type')
        return analytical_sinogram
