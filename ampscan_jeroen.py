# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:07:41 2020

@author: s144339
"""

import csv
import numpy as np
import math
import vtk
import copy
from scipy import spatial
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import basinhopping
from scipy.optimize import brute
from scipy.optimize import fmin
from ampscan.core import AmpObject
from ampscan.vis import vtkRenWin

import operator
import sys
import webbrowser

def generateRegCsv(filename,reg):
    writer = csv.writer(filename)
    for i in reg.reg.values:
        writer.writerow([i])

class customalign(object):
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
            [R, T] = getattr(self, 'optPoint2Point')(mv, sv)
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
        noseindex = np.argmax(self.m.vert[:,2])
        deltax = np.max(self.s.vert[:,0]) - np.max(self.m.vert[:,0])
        deltay = np.max(self.s.vert[:,1]) - self.m.vert[noseindex,1]
        deltaz = np.mean(self.s.vert[:,2]) - self.m.vert[noseindex,2]
        
        self.m.translate([deltax, deltay, deltaz])
    
    @staticmethod
    def optPoint2Point(mv, sv, opt='L-BFGS-B'):
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
        limit = 30  #grid testing limits
        Ns = 6     #number of grid points
        k = 1      #number of grid points chosen to be tested on standard deviation
        n = 3       #number of degrees of freedom
        limpre = ((-np.pi/8, np.pi/8),
                (0, limit),
               (-limit, limit))
        limpost = np.array([[-np.pi/(8*Ns), np.pi/(8*Ns)],
                            [0, limit/Ns],
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
        X = brute(customalign.optDistErrorerr, limpre, args=(mv, sv), Ns=Ns, 
                  full_output=True, finish=None, disp=True)
        print('brute done')
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
            std[i] = customalign.optDistErrorstd(X[1:,Xindex[i]], mv, sv)
        minstd = np.argmin(std)
        
        # Xfinal = minimize(customalign.optDistErrorerr, X[1:,Xindex[minstd]],
        #              args=(mv, sv), bounds=limpost, method=opt, 
        #              options={'disp':True})
        Xfinal = minimize(customalign.optDistErrorerr, X[1:,Xindex[minstd]],
                      args=(mv, sv), method='Nelder-Mead', 
                      options={'disp':True})
        [err, errstd] = customalign.optDistError_end(Xfinal.x, mv, sv)
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
    def optDistErrorerr(X, mv, sv):
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
        
        coor = np.zeros((len(sv[:,0]),np.size(moved,0),3))
        dist = np.zeros((len(sv[:,0]),np.size(moved,0)))
        for i in range(np.size(coor,0)):
            coor[i,:,0] = sv[i,0]*np.ones((1,np.size(moved,0)))
            coor[i,:,1] = sv[i,1]*np.ones((1,np.size(moved,0)))
            coor[i,:,2] = sv[i,2]*np.ones((1,np.size(moved,0)))
        dist = np.linalg.norm(coor[:,:,:] - moved[:,:], axis=2)
        # dist = np.sqrt(np.sum(np.power(coor[:,:,:] - moved[:,:],2),axis=2))
        err = np.mean(np.amin(dist, axis=1))
        # err = np.mean(min(np.sqrt(np.sum([xdist, ydist, zdist], axis=1))))
        
        # err = np.zeros(np.size(sv,0))
        
        # for i in range(np.size(sv,0)):
        #     svi = np.ones((np.size(moved,0),np.size(moved,1)))*sv[i,:]
        #     err[i] = min(np.sqrt(np.sum(np.power((svi-moved),2),axis=1)))
                    
        # err = np.mean(err)
        
        # dist = (sv - moved)**2
        # dist = dist.sum(axis=1)
        # err = np.sqrt(dist.mean())
        print('error:')
        print(err)
        return err
    
    @staticmethod
    def optDistErrorstd(X, mv, sv):
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
        # moved += X[3:]
        moved = moved + np.array([0, X[1], X[2]])
        
        # xdist = np.power((np.dot(sv[:,0],np.ones(np.size(moved,0))) - 
        #                  moved[:,0]),2)
        # xdist = np.dot(sv[:,0],np.ones(np.size(moved,0),1))# - moved[:,0]
        # print(sv[:,0])
        # print(np.ones(np.size(moved,0),1))
        # ydist = np.power(np.dot(sv[:,1],np.ones(np.size(moved,0))) - 
        #                  moved[:,1],2)
        # zdist = np.power(np.dot(sv[:,2],np.ones(np.size(moved,0))) - 
        #                  moved[:,2],2)
        # err = np.mean(min(np.sqrt(np.sum([xdist, ydist, zdist], axis=1))))
        coor = np.zeros((len(sv[:,0]),np.size(moved,0),3))
        dist = np.zeros((len(sv[:,0]),np.size(moved,0)))
        for i in range(np.size(coor,0)):
            coor[i,:,0] = sv[i,0]*np.ones((1,np.size(moved,0)))
            coor[i,:,1] = sv[i,1]*np.ones((1,np.size(moved,0)))
            coor[i,:,2] = sv[i,2]*np.ones((1,np.size(moved,0)))
        dist = np.linalg.norm(coor[:,:,:] - moved[:,:], axis=2)
        # dist = np.sqrt(np.sum(np.power(coor[:,:,:] - moved[:,:],2),axis=2))
        err = np.amin(dist, axis=1)
        errstd = np.std(err)
        err = np.mean(err)
        # err = np.zeros(np.size(sv,0))
        # for i in range(np.size(sv,0)):
        #     svi = np.ones((np.size(moved,0),np.size(moved,1)))*sv[i,:]
        #     err[i] = min(np.sqrt(np.sum(np.power((svi-moved),2),axis=1)))
        # errstd = np.std(err)
        # err = np.mean(err)
        
        # dist = (sv - moved)**2
        # dist = dist.sum(axis=1)
        # err = np.sqrt(dist.mean())
        print('error:')
        print(err)
        print('errorstd:')
        print(errstd)
        return errstd
    
    @staticmethod
    def optDistError_end(X, mv, sv):
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
        # moved += X[3:]
        moved = moved + np.array([0, X[1], X[2]])
        
        # xdist = np.power(np.dot(sv[:,0],np.ones(np.size(moved,0))) - 
        #                  moved[:,0],2)
        # print(np.shape(xdist))
        # ydist = np.power(np.dot(sv[:,1],np.ones(np.size(moved,0))) - 
        #                  moved[:,1],2)
        # zdist = np.power(np.dot(sv[:,2],np.ones(np.size(moved,0))) - 
        #                  moved[:,2],2)
        # err = np.mean(min(np.sqrt(np.sum([xdist, ydist, zdist], axis=1))))
        coor = np.zeros((len(sv[:,0]),np.size(moved,0),3))
        dist = np.zeros((len(sv[:,0]),np.size(moved,0)))
        for i in range(np.size(coor,0)):
            coor[i,:,0] = sv[i,0]*np.ones((1,np.size(moved,0)))
            coor[i,:,1] = sv[i,1]*np.ones((1,np.size(moved,0)))
            coor[i,:,2] = sv[i,2]*np.ones((1,np.size(moved,0)))
        dist = np.linalg.norm(coor[:,:,:] - moved[:,:], axis=2)
        # dist = np.sqrt(np.sum(np.power(coor[:,:,:] - moved[:,:],2),axis=2))
        err = np.amin(dist, axis=1)
        errstd = np.std(err)
        err = np.mean(err)
        # err = np.zeros(np.size(sv,0))
        # for i in range(np.size(sv,0)):
        #     svi = np.ones((np.size(moved,0),np.size(moved,1)))*sv[i,:]
        #     err[i] = min(np.sqrt(np.sum(np.power((svi-moved),2),axis=1)))
        # errstd = np.std(err)
        
        # err = np.mean(err)
        
        # dist = (sv - moved)**2
        # dist = dist.sum(axis=1)
        # err = np.sqrt(dist.mean())

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