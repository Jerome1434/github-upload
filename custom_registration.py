# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:39:39 2020

@author: s144339
"""
import csv
import numpy as np
import math
import vtk
import copy
from scipy import spatial
from scipy.optimize import minimize
from scipy.optimize import brute
from ampscan.core import AmpObject
from ampscan.vis import vtkRenWin

import matplotlib.pyplot as plt

class customregistration(object):
    r"""
    Registration methods between two AmpObject meshes. This function morphs the baseline 
    vertices onto the surface of the target and returns a new AmpObject
    
    Parameters
    ----------
    baseline: AmpObject
    	The baseline AmpObject, the vertices from this will be morphed onto the target
    target: AmpObject
    	The target AmpObject, the shape that the baseline attempts to morph onto
    method: str: default 'point2plane'
    	A string of the method used for registration
    *args:
    	The arguments used for the registration methods
    **kwargs:
    	The keyword arguments used for the registration methods
        
    Returns
    -------
    reg: AmpObject
        The registered AmpObject, the vertices of this are on the surface of the target 
        and it has the same number of vertices and face array as the baseline AmpObject
        Access this accessing the registration.reg 
    
    Examples
    --------
    >>> from ampscan.core import AmpObject
    >>> baseline = AmpObject(basefh)
    >>> target = AmpObject(targfh)
    >>> reg = registration(baseline, target, steps=10, neigh=10, smooth=1).reg
		
    """ 
    def __init__(self, baseline, target, method='point2plane', *args, **kwargs):
        self.b = baseline
        self.t = target
        if method is not None:
            getattr(self, method)(*args, **kwargs)
        
        
    def point2plane(self, steps = 1, neigh = 10, inside = True, subset = None, 
                    scale=None, smooth=1, fixBrim=False, error='norm'):
        r"""
        Point to Plane method for registration between the two meshes 
        
        Parameters
        ----------
        steps: int, default 1
            Number of iterations
        int, default 10
            Number of nearest neighbours to interrogate for each baseline point
        inside: bool, default True
            If True, a barycentric centre check is made to ensure the registered 
            point lines within the target triangle
        subset: array_like, default None
            Indicies of the baseline nodes to include in the registration, default is none so 
            all are used
        scale: float, default None
            If not None scale the baseline mesh to match the target mesh in the z-direction, 
            the value of scale will be used as a plane from which the nodes are scaled.
            Nodes with a higher z value will not be scaled. 
        smooth: int, default 1
            Indicate number of Laplacian smooth steps in between the steps 
        fixBrim: bool, default False
            If True, the nodes on the brim line will not be included in the smooth
        error: bool, default False
            If True, the polarity will be included when calculating the distance 
            between the target and baseline mesh
		
        """
        # Calc FaceCentroids
        fC = self.t.vert[self.t.faces].mean(axis=1)
        # Construct knn tree
        tTree = spatial.cKDTree(fC)
        bData = dict(zip(['vert', 'faces', 'values'], 
                         [self.b.vert, self.b.faces, self.b.values]))
        regData = copy.deepcopy(bData)
        self.reg = AmpObject(regData, stype='reg')
        self.disp = AmpObject({'vert': self.reg.vert,
                               'faces': self.reg.faces,
                               'values':self.reg.values})
        if scale is not None:
            tmin = self.t.vert.min(axis=0)[2]
            rmin = self.reg.vert.min(axis=0)[2]
            SF = ((tmin-scale)/(rmin-scale)) - 1
            logic = self.reg.vert[:, 2] < scale
            d = (self.reg.vert[logic, 2] - scale) * SF
            self.disp.vert[logic, 2] += d
            self.reg.vert = self.b.vert + self.disp.vert
        normals = np.cross(self.t.vert[self.t.faces[:,1]] -
                         self.t.vert[self.t.faces[:,0]],
                         self.t.vert[self.t.faces[:,2]] -
                         self.t.vert[self.t.faces[:,0]])
        mag = (normals**2).sum(axis=1)
        for step in np.arange(steps, 0, -1, dtype=float):
            # Index of 10 centroids nearest to each baseline vertex
            ind = tTree.query(self.reg.vert, neigh)[1]
            # Define normals for faces of nearest faces
            norms = normals[ind]
            # Get a point on each face
            fPoints = self.t.vert[self.t.faces[ind, 0]]
            # Calculate dot product between point on face and normals
            d = np.einsum('ijk, ijk->ij', norms, fPoints)
            t = (d - np.einsum('ijk, ik->ij', norms, self.reg.vert))/mag[ind]
            # Calculate the vector from old point to new point
            G = self.reg.vert[:, None, :] + np.einsum('ijk, ij->ijk', norms, t)
            # Ensure new points lie inside points otherwise set to 99999
            # Find smallest distance from old to new point 
            if inside is False:
                G = G - self.reg.vert[:, None, :]
                GMag = np.sqrt(np.einsum('ijk, ijk->ij', G, G))
                GInd = GMag.argmin(axis=1)
            else:
                G, GInd = self.__calcBarycentric(self.reg.vert, G, ind)
            # Define vector from baseline point to intersect point
            D = G[np.arange(len(G)), GInd, :]
#            rVert += D/step
            self.disp.vert += D/step
            if smooth > 0 and step > 1:
                self.disp.lp_smooth(smooth, brim = fixBrim)
                self.reg.vert = self.b.vert + self.disp.vert
            else:
                self.reg.vert = self.b.vert + self.disp.vert
                self.reg.calcNorm()
        self.reg.calcStruct(vNorm=True)
        self.reg.values[:] = self.calcError(error)
    
    def point2point(self, steps = 1, neigh = 10, error='norm'):
        r"""
        Direct minimisation of the rmse between the points of the two meshes. This 
        method enables access to all of Scipy's minimisation algorithms 
        
        Returns
        -------
        R: ndarray
            The optimal rotation array 
        T: ndarray
            The optimal translation array
            
        Examples
        --------
        >>> static = AmpObject(staticfh)
        >>> moving = AmpObject(movingfh)
        >>> al = align(moving, static, method='optPoint2Point', opt='SLSQP').m
            
        Parameters
        ----------
        steps: int, default 1
            Number of iterations
        int, default 10
            Number of nearest neighbours to interrogate for each baseline point
        error: bool, default False
            If True, the polarity will be included when calculating the distance 
            between the target and baseline mesh
		
        """
        # Calc FaceCentroids
        fC = self.t.vert
        # Construct knn tree
        self.t.tTree = spatial.cKDTree(fC)
        bData = dict(zip(['vert', 'faces', 'values'], 
                         [self.b.vert, self.b.faces, self.b.values]))
        regData = copy.deepcopy(bData)
        self.reg = AmpObject(regData, stype='reg')
        self.disp = AmpObject({'vert': np.zeros(self.reg.vert.shape),
                               'faces': self.reg.faces,
                               'values':self.reg.values})

#         normals = np.cross(self.t.vert[self.t.faces[:,1]] -
#                          self.t.vert[self.t.faces[:,0]],
#                          self.t.vert[self.t.faces[:,2]] -
#                          self.t.vert[self.t.faces[:,0]])
#         mag = (normals**2).sum(axis=1)
#         for step in np.arange(steps, 0, -1, dtype=float):
#             # Index of 10 centroids nearest to each baseline vertex
#             ind = tTree.query(self.reg.vert, neigh)[1]
#             # Define normals for faces of nearest faces
#             norms = normals[ind]
#             # Get a point on each face
#             fPoints = self.t.vert[self.t.faces[ind, 0]]
#             # Calculate dot product between point on face and normals
#             d = np.einsum('ijk, ijk->ij', norms, fPoints)
#             t = (d - np.einsum('ijk, ik->ij', norms, self.reg.vert))/mag[ind]
#             # Calculate the vector from old point to new point
#             G = self.reg.vert[:, None, :] + np.einsum('ijk, ij->ijk', norms, t)
#             # Ensure new points lie inside points otherwise set to 99999
#             # Find smallest distance from old to new point 
#             G = G - self.reg.vert[:, None, :]
#             GMag = np.sqrt(np.einsum('ijk, ijk->ij', G, G))
#             GInd = GMag.argmin(axis=1)

#             # Define vector from baseline point to intersect point
#             D = G[np.arange(len(G)), GInd, :]
# #            rVert += D/step
#             self.disp.vert += D/step
            
#             self.reg.vert = self.b.vert + self.disp.vert
#             self.reg.calcNorm()
        self.reg.calcNorm()
        self.reg.calcStruct(vNorm=True)
        
        error = 'customnorm'
        self.reg.values[:] = self.calcError(error)
    
    def calcError(self, method='norm'):
        r"""
        Calculate the magnitude of distances between the baseline and registered array
		
        Parameters
        ----------
        method: str, default 'norm'
            The method used to calculate the distances. 'abs' returns the absolute
            distance. 'cent'calculates polarity based upon distance from centroid.
            'norm' calculates dot product between baseline vertex normal and distance 
            normal

        Returns
        -------
        values: array_like
            Magnitude of distances

        """
        method = '_customregistration__' + method + 'Dist'
        
        # try:
        values = getattr(self, method)()
        return values
        # except: 
        #     ValueError('"%s" is not a method, try "abs", "cent" or "prod"' % method)

    def __absDist(self):
        r"""
        Return the error based upon the absolute distance
        
        Returns
        -------
        values: array_like
            Magnitude of distances

        """
        print('absdist')
        return np.linalg.norm(self.reg.vert - self.b.vert, axis=1)
    
    def __centDist(self):
        r"""
        Return the error based upon distance from centroid 
        
        Returns
        -------
        values: array_like
            Magnitude of distances

        """
        values = np.linalg.norm(self.reg.vert - self.b.vert, axis=1)
        cent = self.b.vert.mean(axis=0)
        r = np.linalg.norm(self.reg.vert - cent, axis=1)
        b = np.linalg.norm(self.b.vert - cent, axis=1)
        polarity = np.ones([self.reg.vert.shape[0]])
        polarity[r<b] = -1
        print(values)
        return values * polarity

    def __normDist(self):
        r"""
        Returns error based upon scalar product of normal 
        
        Returns
        -------
        values: array_like
            Magnitude of distances

        """
        self.b.calcVNorm()
        D = self.reg.vert - self.b.vert
        n = self.b.vNorm
        values = np.linalg.norm(D, axis=1)
        polarity = np.sum(n*D, axis=1) < 0
        values[polarity] *= -1.0
        print(values)
        return values
    
    def __customnormDist(self):
        r"""
        Returns error based upon scalar product of normal 
        
        Returns
        -------
        values: array_like
            Magnitude of distances

        """
        
        [dist, idx] = self.t.tTree.query(self.reg.vert, 1)
        
        tData = dict(zip(['vert', 'faces', 'values'], 
                         [self.t.vert, self.t.faces, self.t.values]))
        tData = copy.deepcopy(tData)
        self.customt = AmpObject(tData, stype='limb')
        self.customt.calcVNorm()
        
        D = self.reg.vert - self.customt.vert[idx]
        n = self.customt.vNorm[idx]
        values = np.linalg.norm(D, axis=1)
        
        polarity = np.sum(n*D, axis=1) < 0
        values[polarity] *= -1.0
        return values
    
    def plotResults(self, name=None, xrange=None, color=None, alpha=None):
        r"""
        Function to generate a mpl figure. Includes a rendering of the 
        AmpObject, a histogram of the registration values 
        
        Returns
        -------
        fig: mplfigure
            A matplot figure of the standard analysis
        
        """
        fig, ax = plt.subplots(1)
        n, bins, _ = ax.hist(self.reg.values, 50, density=True, range=xrange,
                             color=color, alpha=alpha)
        mean = self.reg.values.mean()
        stdev = self.reg.values.std()
        ax.set_title(r'Distribution of shape variance, '
                     '$\mu=%.2f$, $\sigma=%.2f$' % (mean, stdev))
        ax.set_xlim(None)
        if name is not None:
            plt.savefig(name, dpi = 300)
        return ax, n, bins
    
    def customvNorm(self):
        """
        Function to compute the vertex normals based upon the mean of the
        connected face normals 
        
        Returns
        -------
        vNorm: ndarray
            normal of each vertex

        """
        f = self.faces.flatten()
        o_idx = f.argsort()
        row, col = np.unravel_index(o_idx, self.faces.shape)
        ndx = np.searchsorted(f[o_idx], range(self.vert.shape[0]), side='right')
        ndx = np.r_[0, ndx]
        norms = self.norm[row, :]
        self.vNorm = np.zeros(self.vert.shape)
        for i in range(self.vert.shape[0]):
            self.vNorm[i, :] = np.nanmean(norms[ndx[i]:ndx[i+1], :], axis=0)
    
def generateRegCsv(filename,reg):
    with open(filename, 'w', newline='') as myfile:
        writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for i in reg.reg.values:
            writer.writerow([i])