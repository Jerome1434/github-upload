# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:07:41 2020

@author: s144339
"""

import numpy as np
import math
import vtk
import copy
from scipy import spatial
from scipy.optimize import minimize
from scipy.optimize import brute
from ampscan.core import AmpObject
from ampscan.vis import vtkRenWin

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
    
    def __init__(self, base, target, method='best-fit'):
        '''
        base : 
            the base object, which is static
        target : 
            the target object, which will be moved. 
        
        The error will be calculated based on all values for the minimum 
        distance between one base vertex(face centroid) and all target vertices.
        
        method : 
            the method distinguishes between a best-fit algorithm and a 
            best-fit algorithm including that the whole mask is located inside
            the face, to be able to calculate the max. distance
            'best-fit' for the standard best-fit algorithm
            'max-dist' for the max. distance best-fit algorithm
        '''
        mData = dict(zip(['vert', 'faces', 'values'], 
                         [target.vert, target.faces, target.values]))
        alData = copy.deepcopy(mData)
        self.m = AmpObject(alData, stype='reg')
        self.s = base
        self.initialize()
        self.runcustomICP(maxiter=2, inlier=1.0, initTransform=None, method=method)
        
    def runcustomICP(self, maxiter=20, inlier=1.0, initTransform=None, method='best-fit'):
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
        self.s.kdTree = spatial.cKDTree(self.s.vert)
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
        # sv = fC[idx, :]
        
        self.s.calcVNorm()
        
        for i in range(1,maxiter):
            [R, T] = getattr(self, 'optPoint2Point')(mv, kdTree, self.s, 
                                                     self.m, method=method)
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
        
        deltax = self.s.vert[noseindex,0] - np.mean(self.m.vert[:,0])
        deltay = self.s.vert[noseindex,1] - np.max(self.m.vert[:,1]) + 40
        deltaz = self.s.vert[noseindex,2] - np.mean(self.m.vert[:,2]) - 20
        self.m.translate([deltax, deltay, deltaz])
    
    @staticmethod
    def optPoint2Point(mv, kdTree, static, moving, opt='L-BFGS-B', 
                       method='best-fit'):
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
        limit = 15  #grid testing limits
        Ns = 5     #number of grid points
        k = 5      #number of grid points chosen to be tested on standard deviation
        n = 5       #number of degrees of freedom
        limpre = ((-np.pi/16, np.pi/16),
                  (-np.pi/16, np.pi/16),
                  (-limit, limit),
                  (-limit, limit),
                  (-limit, limit))
        
        X1 = brute(customalign.optDistError, limpre, args=(mv, kdTree, 
                   'err'), Ns=Ns, full_output=True, finish=None, disp=True)

        print('brute done')
        print(X1[0])
        print(X1[1])
        print(customalign.con(X1[0], static, moving))
        # X[2] is all tested variables, access by: X[2][0]
        # X[3] is the output
        Xrotres = np.reshape(X1[2][0],[1,Ns**n])   
        Zrotres = np.reshape(X1[2][1],[1,Ns**n])   
        Xres = np.reshape(X1[2][2],[1,Ns**n])
        Yres = np.reshape(X1[2][3],[1,Ns**n])
        Zres = np.reshape(X1[2][4],[1,Ns**n])
        res = np.reshape(X1[3],[1,Ns**n])
        X2 = np.array([res[0,:], Xrotres[0,:], Zrotres[0,:], Xres[0,:], 
                       Yres[0,:], Zres[0,:]])
        idx = []
        if method == 'max-dist':
            for i in range(X2.shape[1]):
                constraint = customalign.con(X2[1:,i], static, moving)
                if constraint < 0:
                    idx.append(i)
            np.delete(X2, idx, axis=0)
            
        Xindex = np.argpartition(X2[0,:], k)[:k]
        Xfinal = False
        for i in range(k):
            limpost = np.array([[X2[1,Xindex[i]] - np.pi/(16*Ns), 
                                 X2[1,Xindex[i]] + np.pi/(16*Ns)],
                                [X2[2,Xindex[i]] - np.pi/(16*Ns), 
                                 X2[2,Xindex[i]] + np.pi/(16*Ns)],
                                [X2[3,Xindex[i]] - limit/Ns, 
                                 X2[3,Xindex[i]] + limit/Ns],
                                [X2[4,Xindex[i]] - limit/Ns, 
                                 X2[4,Xindex[i]] + limit/Ns],
                                [X2[5,Xindex[i]] - limit/Ns, 
                                 X2[5,Xindex[i]] + limit/Ns]])
            if method == 'best-fit':
                X3 = minimize(customalign.optDistError, X2[1:,Xindex[i]],
                              args=(mv, kdTree, 'err'), method='SLSQP', 
                              bounds=limpost, options={'disp':True})
            if method == 'max-dist':
                X3 = minimize(customalign.optDistError, X2[1:,Xindex[i]],
                              args=(mv, kdTree, 'err'), method='SLSQP', 
                              bounds=limpost, 
                              constraints={'type':'ineq', 
                              'fun':customalign.con, 'args':(static, moving)}, 
                              options={'disp':True})
                print(customalign.con(X3.x, static, moving))
            if Xfinal == False:
                Xfinal = X3
            else:
                if X3.fun < Xfinal.fun:
                    Xfinal = X3

        [err, errstd] = customalign.optDistError(Xfinal.x, mv, kdTree, 'end')
        print(Xfinal)
        print([err, errstd])
        
        Xfinal = np.array([Xfinal.x[0], 0, Xfinal.x[1], Xfinal.x[2], 
                           Xfinal.x[3], Xfinal.x[4]])
        [angx, angy, angz] = Xfinal[:3]

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

        [angx, angy, angz] = np.array([X[0], 0, X[1]])
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
        moved = moved + np.array([X[2], X[3], X[4]])
        
        [dist, idx] = kdTree.query(moved, 1)
        dist = np.absolute(dist)
        errstd = np.std(dist)
        err = np.mean(dist)

        if out == 'err':
            return err
        elif out == 'std':
            return errstd
        elif out == 'end':
            print('error:')
            print(err)
            print('standard deviation:')
            print(errstd)
            return [err, errstd]
    
    @staticmethod
    def con(X, static, moving):
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
        
        [angx, angy, angz] = np.array([X[0], 0, X[1]])
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
        moving.vert = np.dot(moving.vert, R.T)
        moving.vert = moving.vert + np.array([X[2], X[3], X[4]])
        
        [dist, idx] = static.kdTree.query(moving.vert, 1)
        
        D = static.vert[idx] - moving.vert
        n = static.vNorm[idx]
        values = np.linalg.norm(D, axis=1)
        
        polarity = np.sum(n*D, axis=1) < 0
        values[polarity] *= -1.0
        
        return -np.max(values)
        
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