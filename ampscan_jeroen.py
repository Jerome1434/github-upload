# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:07:41 2020

@author: s144339
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import math
import vtk
import copy
from scipy import spatial
from scipy.optimize import minimize
from scipy.optimize import brute
from ampscan.core import AmpObject
from ampscan.vis import vtkRenWin

def generateRegCsv(filename,reg):
    writer = csv.writer(filename)
    for i in reg.reg.values:
        writer.writerow([i])

class customalign(object):
    r"""
    Automated alignment methods between two meshes
    
    Parameters
    ----------
    moving: AmpObject
        The moving AmpObject that is to be aligned to the static object
    static: AmpObject
        The static AmpObject that the moving AmpObject that the moving object 
        will be aligned to
    method: str, default 'linPoint2Plane'
        A string of the method used for alignment
    *args:
    	The arguments used for the registration methods
    **kwargs:
    	The keyword arguments used for the registration methods

    Returns
    -------
    m: AmpObject
        The aligned AmpObject, it same number of vertices and face array as 
        the moving AmpObject
        Access this using align.m

    Examples
    --------
    >>> static = AmpObject(staticfh)
    >>> moving = AmpObject(movingfh)
    >>> al = align(moving, static).m

    """    
    
    def __init__(self, base, target):
        mData = dict(zip(['vert', 'faces', 'values'], [target.vert, target.faces, target.values]))
        alData = copy.deepcopy(mData)
        self.m = AmpObject(alData, stype='reg')
        self.s = base
        self.initialize()
        
        self.runcustomICP(maxiter=2, inlier=1.0, initTransform=None)
        
    def runcustomICP(self, maxiter=20, inlier=1.0, initTransform=None):
        #copied from runICP
        # Define the rotation, translation, error and quaterion arrays
        Rs = np.zeros([3, 3, maxiter+1])
        Ts = np.zeros([3, maxiter+1])
#        qs = np.r_[np.ones([1, maxiter+1]), 
#                   np.zeros([6, maxiter+1])]
#        dq  = np.zeros([7, maxiter+1])
#        dTheta = np.zeros([maxiter+1])
        err = np.zeros([maxiter+1])
        if initTransform is None:
            initTransform = np.eye(4)
        Rs[:, :, 0] = initTransform[:3, :3]
        Ts[:, 0] = initTransform[3, :3]
#        qs[:4, 0] = self.rot2quat(Rs[:, :, 0]) 
#        qs[4:, 0] = Ts[:, 0]
        # Define 
        fC = self.s.vert[self.s.faces].mean(axis=1)
        kdTree = spatial.cKDTree(fC)
        self.m.rigidTransform(Rs[:, :, 0], Ts[:, 0])
        inlier = math.ceil(self.m.vert.shape[0]*inlier)
        [dist, idx] = kdTree.query(self.m.vert, 1)
        # Sort by distance
        sort = np.argsort(dist)
        # Keep only those within the inlier fraction
        [dist, idx] = [dist[sort], idx[sort]]
        [dist, idx, sort] = dist[:inlier], idx[:inlier], sort[:inlier]
        err[0] = math.sqrt(dist.mean())
        
        mv = self.m.vert[sort, :]
        sv = fC[idx, :]
        
        for i in range(1,maxiter):
            [R, T] = getattr(self, 'optPoint2Point')(mv, kdTree)
            print(R)
            print(T)
            Rs[:, :, i+1] = np.dot(R, Rs[:, :, i])
            Ts[:, i+1] = np.dot(R, Ts[:, i]) + T
            self.m.rigidTransform(R, T)
            [dist, idx] = kdTree.query(self.m.vert, 1)
            sort = np.argsort(dist)
            [dist, idx] = [dist[sort], idx[sort]]
            [dist, idx, sort] = dist[:inlier], idx[:inlier], sort[:inlier]
            err[i+1] = math.sqrt(dist.mean())
        R = Rs[:, :, -1]
        #Simpl
        [U, s, V] = np.linalg.svd(R)
        R = np.dot(U, V)
        self.tForm = np.r_[np.c_[R, np.zeros(3)], np.append(Ts[:, -1], 1)[:, None].T]
        self.R = R
        self.T = Ts[:, -1]
        self.rmse = err[-1]
        print('rmse:')
        print(self.rmse)
    
    def initialize(self):
        noseindex = np.argmax(self.s.vert[:,2])
        deltax = np.max(self.s.vert[:,0]) - np.max(self.m.vert[:,0])
        deltay = self.s.vert[noseindex,1] - np.max(self.m.vert[:,1]) + 40
        deltaz = self.s.vert[noseindex,2] - np.mean(self.m.vert[:,2]) - 20
        
        self.m.translate([deltax, deltay, deltaz])
    
    @staticmethod
    def optPoint2Point(mv, kdTree, opt='L-BFGS-B'):
        r"""
        Direct minimisation of the rmse between the points of the two meshes. This 
        method enables access to all of Scipy's minimisation algorithms 
        
        Parameters
        ----------
        mv: ndarray
            The array of vertices to be moved 
        sv: ndarray
            The array of static vertices, these are the face centroids of the 
            static mesh
        opt: str, default 'L_BFGS-B'
            The string of the scipy optimiser to use 
        
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
            
        """
        # opt = 'L-BFGS-B'
        # X = np.zeros(6)
        # lim = [0, 0] * 3 + [None, None] * 3
        # lim = np.reshape(lim, [6, 2])
        limit = 10  #grid testing limits
        Ns = 6     #number of grid points
        k = 1      #number of grid points chosen to be tested on standard deviation
        n = 3       #number of degrees of freedom
        limpre = ((-np.pi/16, np.pi/16),
                (-limit, limit),
               (-limit, limit))
        limpost = np.array([[-np.pi/(16*Ns), np.pi/(16*Ns)],
                            [-limit/Ns, limit/Ns],
                            [-limit/Ns, limit/Ns]])
        X = np.zeros(n)
        # try:
            
        # X = minimize(customalign.optDistError, X,args=(mv, sv),bounds=lim, 
        #              method=opt, options={'disp':True})
        # X = minimize(customalign.optDistError, X,args=(mv, sv), bounds=lim,
        #               method='TNC', options={'disp':True})
        # minimizerkwargs = {'args':(mv, sv), 'method':'L-BFGS-B', 'bounds':lim}
        
        # #new loop:
        # counter = 1
        # err = 1
        # errstd = 6
        # while (err > 0 and errstd > 2) or counter < 20: 
        #     if counter%2 != 0:
        #         X = np.zeros(3)
        #         Xopt = basinhopping(customalign.optDistErrorerr, X, niter = 20,
        #                          minimizer_kwargs = minimizerkwargs, disp=True, 
        #                          niter_success=(3+counter))
        #         X = Xopt.x
        #     else:
        #         Xopt = minimize(customalign.optDistErrorstd, X, args=(mv,sv), 
        #                      bounds=lim, method = opt, options={'disp':True})
        #         X = Xopt.x
        #     counter += 1
            
        #     [err, errstd] = customalign.optDistError_end(X, mv, sv)
        #     print('error:')
        #     print(err)
        #     print('errorstd:')
        #     print(errstd)
        X = brute(customalign.optDistError, limpre, args=(mv, kdTree, 'err'), Ns=Ns, 
                  full_output=True, finish=None, disp=True)
        print('brute done')
        print(X[0])
        print(X[1])
        #X[2] is all tested variables, access by: X[2][0]
        #X[3] is the output
        Xrotres = np.reshape(X[2][0],[1,Ns**n])   
        Yres = np.reshape(X[2][1],[1,Ns**n])
        Zres = np.reshape(X[2][2],[1,Ns**n])
        res = np.reshape(X[3],[1,Ns**n])
        X = np.array([res[0,:], Xrotres[0,:], Yres[0,:], Zres[0,:]])
        # print('np.amin(X[0,:])')
        # print(np.amin(X[0,:]))
        Xindex = np.argpartition(X[0,:], k)[:k]
        
        std = np.zeros(k)
        for i in range(0,k-1):
            std[i] = customalign.optDistError(X[1:,Xindex[i]], mv, kdTree, 'std')
        minstd = np.argmin(std)
        
        # Xfinal = minimize(customalign.optDistErrorerr, X[1:,Xindex[minstd]],
        #              args=(mv, sv), bounds=limpost, method=opt, 
        #              options={'disp':True})
        Xfinal = minimize(customalign.optDistError, X[1:,Xindex[minstd]],
                      args=(mv, kdTree, 'err'), method=opt, bounds=limpost,
                      options={'disp':True})
        [err, errstd] = customalign.optDistError(Xfinal.x, mv, kdTree, 'end')
        print(Xfinal)
        print([err, errstd])
        # print(X2)
        # print(lim2)
        # lim2 = np.array([lim2,
        #                  [-limit/Ns, -limit/Ns],
        #                  [-limit/Ns, -limit/Ns]])
        # X2[1:] = X1
        # X2 = minimize(customalign.optDistErrorstd, X2, args=(mv, sv),
        #               bounds=lim2, method='TNC', options={'disp':True})
        
        Xfinal = np.array([Xfinal.x[0], 0, 0, 0, Xfinal.x[1], Xfinal.x[2]])
        [angx, angy, angz] = Xfinal[:3]
        # except:
        #     X = minimize(customalign.optDistError, X,
        #                   args=(mv, sv),
        #                   method=opt)
        
        # [angx, angy, angz] = X.x[:3]
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angx), -np.sin(angx)],
                       [0, np.sin(angx), np.cos(angx)]])
        Ry = np.array([[np.cos(angy), 0, np.sin(angy)],
                       [0, 1, 0],
                       [-np.sin(angy), 0, np.cos(angy)]])
        Rz = np.array([[np.cos(angz), -np.sin(angz), 0],
                       [np.sin(angz), np.cos(angz), 0],
                       [0, 0, 1]])
        R = np.dot(np.dot(Rz, Ry), Rx)
        # T = X.x[3:]
        T = Xfinal[3:]
        return (R, T)
    
    @staticmethod
    def optDistError(X, mv, kdTree, out='err'):
        r"""
        The function to minimise. It performs the affine transformation then returns 
        the rmse between the two vertex sets
        
        Parameters
        ----------
        X:  ndarray
            The affine transformation corresponding to [Rx, Ry, Rz, Tx, Ty, Tz]
        mv: ndarray
            The array of vertices to be moved 
        sv: ndarray
            The array of static vertices, these are the face centroids of the 
            static mesh
        out:string
            'err', 'std', or 'end'. determines which output is returned

        Returns
        -------
        err: float
            The RMSE between the two meshes
        
        """

        [angx, angy, angz] = [X[0], 0, 0]
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(angx), -np.sin(angx)],
                        [0, np.sin(angx), np.cos(angx)]])
        Ry = np.array([[np.cos(angy), 0, np.sin(angy)],
                        [0, 1, 0],
                        [-np.sin(angy), 0, np.cos(angy)]])
        Rz = np.array([[np.cos(angz), -np.sin(angz), 0],
                        [np.sin(angz), np.cos(angz), 0],
                        [0, 0, 1]])
        R = np.dot(np.dot(Rz, Ry), Rx)
        moved = np.dot(mv, R.T)
        moved = moved + np.array([0, X[1], X[2]])
        
        [dist, idx] = kdTree.query(moved, 1)
        dist = np.absolute(dist)
        # print(dist)
        # coor = np.zeros((len(sv[:,0]),np.size(moved,0),3))
        # dist = np.zeros((len(sv[:,0]),np.size(moved,0)))
        # for i in range(np.size(coor,0)):
        #     coor[i,:,0] = sv[i,0]*np.ones((1,np.size(moved,0)))
        #     coor[i,:,1] = sv[i,1]*np.ones((1,np.size(moved,0)))
        #     coor[i,:,2] = sv[i,2]*np.ones((1,np.size(moved,0)))
        # dist = np.linalg.norm(coor[:,:,:] - moved[:,:], axis=2)
        errstd = np.std(dist)
        err = np.mean(dist)

        if out == 'err':
            print('error:')
            print(err)
            return err
        elif out == 'std':
            print('standard deviation:')
            print(errstd)
            return errstd
        elif out == 'end':
            print('error:')
            print(err)
            print('standard deviation:')
            print(errstd)
            return [err, errstd]
    
    def display(self):
        r"""
        Display the static mesh and the aligned within an interactive VTK 
        window 
        
        """
        if not hasattr(self.s, 'actor'):
            self.s.addActor()
        if not hasattr(self.m, 'actor'):
            self.m.addActor()
        # Generate a renderer window
        win = vtkRenWin()
        # Set the number of viewports
        win.setnumViewports(1)
        # Set the background colour
        win.setBackground([1,1,1])
        # Set camera projection 
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(win)
        renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        # Set camera projection 
        win.setView()
        self.s.actor.setColor([1.0, 0.0, 0.0])
        self.s.actor.setOpacity(0.5)
        self.m.actor.setColor([0.0, 0.0, 1.0])
        self.m.actor.setOpacity(0.5)
        win.renderActors([self.s.actor, self.m.actor])
        win.Render()
        win.rens[0].GetActiveCamera().Azimuth(180)
        win.rens[0].GetActiveCamera().SetParallelProjection(True)
        win.Render()
        return win
    
    def genIm(self, crop=False):
        r"""
        Display the static mesh and the aligned within an interactive VTK 
        window 
        
        """
        if not hasattr(self.s, 'actor'):
            self.s.addActor()
        if not hasattr(self.m, 'actor'):
            self.m.addActor()
        # Generate a renderer window
        win = vtkRenWin()
        # Set the number of viewports
        win.setnumViewports(1)
        # Set the background colour
        win.setBackground([1,1,1])
        # Set camera projection 
        # Set camera projection 
        win.setView([0, -1, 0], 0)
        win.SetSize(512, 512)
        win.Modified()
        win.OffScreenRenderingOn()
        self.s.actor.setColor([1.0, 0.0, 0.0])
        self.s.actor.setOpacity(0.5)
        self.m.actor.setColor([0.0, 0.0, 1.0])
        self.m.actor.setOpacity(0.5)
        win.renderActors([self.s.actor, self.m.actor])
        win.Render()
        win.rens[0].GetActiveCamera().Azimuth(0)
        win.rens[0].GetActiveCamera().SetParallelProjection(True)
        win.Render()
        im = win.getImage()
        if crop is True:
            mask = np.all(im == 1, axis=2)
            mask = ~np.all(mask, axis=1)
            im = im[mask, :, :]
            mask = np.all(im == 1, axis=2)
            mask = ~np.all(mask, axis=0)
            im = im[:, mask, :]
        return im, win

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
        self.disp = AmpObject({'vert': np.zeros(self.reg.vert.shape),
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
        method = '_registration__' + method + 'Dist'
        try:
            values = getattr(self, method)()
            return values
        except: ValueError('"%s" is not a method, try "abs", "cent" or "prod"' % method)
        

    
    def __absDist(self):
        r"""
        Return the error based upon the absolute distance
        
        Returns
        -------
        values: array_like
            Magnitude of distances

        """
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
        return values
        
        
    def __calcBarycentric(self, vert, G, ind):
        r"""
        Calculate the barycentric co-ordinates of each target face and the registered vertex, 
        this ensures that the registered vertex is within the bounds of the target face. If not 
        the registered vertex is moved to the nearest vertex on the target face 

        Parameters
        ----------
        vert: array_like
            The array of baseline vertices
        G: array_like
            The array of candidates for registered vertices. If neigh>1 then axis 2 will correspond 
            to the number of nearest neighbours selected
        ind: array_like
            The index of the nearest faces to the baseline vertices
        
        Returns
        -------
        G: array_like 
            The new array of candidates for registered vertices, from here, the one with 
            smallest magnitude is selected. All these points will lie within the target face
        GInd: array_like
            The index of the shortest distance between each baseline vertex and the registered vertex
            
        """
        P0 = self.t.vert[self.t.faces[ind, 0]]
        P1 = self.t.vert[self.t.faces[ind, 1]]
        P2 = self.t.vert[self.t.faces[ind, 2]]
        
        v0 = P2 - P0
        v1 = P1 - P0
        v2 = G - P0
        
        d00 = np.einsum('ijk, ijk->ij', v0, v0)
        d01 = np.einsum('ijk, ijk->ij', v0, v1)
        d02 = np.einsum('ijk, ijk->ij', v0, v2)
        d11 = np.einsum('ijk, ijk->ij', v1, v1)
        d12 = np.einsum('ijk, ijk->ij', v1, v2)
        
        denom = d00*d11 - d01*d01
        u = (d11 * d02 - d01 * d12)/denom
        v = (d00 * d12 - d01 * d02)/denom
        # Test if inside 
        logic = (u >= 0) * (v >= 0) * (u + v < 1)
        
        P = np.stack([P0, P1, P2], axis=3)
        pg = G[:, :, :, None] - P
        pd =  np.linalg.norm(pg, axis=2)
        pdx = pd.argmin(axis=2)
        i, j = np.meshgrid(np.arange(P.shape[0]), np.arange(P.shape[1]))
        nearP = P[i.T, j.T, :, pdx]
        G[~logic, :] = nearP[~logic, :]
        G = G - vert[:, None, :]
        GMag = np.sqrt(np.einsum('ijk, ijk->ij', G, G))
        GInd = GMag.argmin(axis=1)
        return G, GInd
    
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