# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:32:53 2020

@author: s144339
"""


import sys
import webbrowser

import os
import copy
import csv
from scipy import spatial
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.colorbar as clb

import math
import numpy as np
import vtk
from ampscan import AmpObject, analyse
from ampscan.registration import registration
from ampscan.align import align
from ampscan.vis import qtVtkWindow, vtkRenWin

from PyQt5.QtCore import QPoint, QSize, Qt, QTimer, QRect, pyqtSignal
from PyQt5.QtGui import (QColor, QFontMetrics, QImage, QPainter, QIcon,
                         QOpenGLVersionProfile)
from PyQt5.QtWidgets import (QAction, QApplication, QGridLayout, QHBoxLayout,
                             QMainWindow, QMessageBox, QComboBox, QButtonGroup,
                             QOpenGLWidget, QFileDialog, QLabel, QPushButton,
                             QSlider, QWidget, QTableWidget, QTableWidgetItem,
                             QAbstractButton, QCheckBox, QErrorMessage, QTextEdit, QVBoxLayout)


class ampscanGUI(QMainWindow):
    """
    Generates an GUI for handling stl data. Window is derived from QT.


    Example
    -------
    Perhaps an example implementation:

    >>> from GUIs.ampscanGUI import ampscanGUI

    """

    def __init__(self, parent = None):
        super(ampscanGUI, self).__init__()
        self.vtkWidget = qtVtkWindow()
        self.renWin = self.vtkWidget._RenderWindow
        self.renWin.setBackground([1,1,1])
        self.mainWidget = QWidget()
        self.files = {}
        self.filesDrop = list(self.files.keys())
#        self.CMap = np.array([[212.0, 221.0, 225.0],
#                              [31.0, 73.0, 125.0]])/255.0
        self.setCentralWidget(self.mainWidget)
        self.createActions()
        self.createMenus()
        self.Layout = QGridLayout()
        self.Layout.addWidget(self.vtkWidget, 0, 0)
        self.mainWidget.setLayout(self.Layout)
        self.setWindowTitle("ampscan")
        self.resize(800, 800)
        self.show()
        self.fileManager = fileManager(self)
        self.fileManager.show()
        self.fileManager.table.itemChanged.connect(self.display)
        self.pnt = None
        self.AmpObj = None

    def chooseOpenFile(self):
        """
        Handles importing of stls into the GUI.

        
        """
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            filter="Meshes (*.stl)")
        if fname[0] == '':
            return
        name = fname[0][:-4].split('/')[-1]
        self.files[name] = AmpObject(fname[0], 'limb')
        amp = self.files[name]
        amp.addActor()
        amp.tform = vtk.vtkTransform()
        amp.tform.PostMultiply()
        amp.actor.SetUserTransform(amp.tform)
#        amp.centre()
        self.fileManager.addRow(name, amp)
        self.display()
        self.filesDrop.append(name)
        if hasattr(self, 'alCont'):
            self.alCont.getNames()
        if hasattr(self, 'regCont'):
            self.regCont.getNames()
#        self.AmpObj.lp_smooth()

    def saveFile(self):
        self.saveCont = saveControls(self.filesDrop, self)
        self.saveCont.show()
        self.saveCont.save.clicked.connect(self.chooseSaveFile)

    def chooseSaveFile(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file',
                                            filter="Meshes (*.stl)")
        if fname[0] == '':
            return
        file = str(self.saveCont.files.currentText())
        self.files[file].save(fname[0])
        try:
            f = open(fname[0]+'.txt','w+')
            f.write('{}'.format(self.pnt))
        except AttributeError:
            print('A point has not been selected')

    def display(self):
        render = []

        for r in range(self.fileManager.n):
            [name, _, color, opacity, display] = self.fileManager.getRow(r)
            # Make the object visible
            if display == 2:
                # TODO fix duplicate names
                if name in self.files.keys():
                    render.append(self.files[name].actor)
                else:
                    show_message("Invalid name: {}".format(name))
                    # Temp workaround name change crash
                    # TODO make names actually change
                    continue

            # Change the color
            try:
                color = color[1:-1].split(',')
                color = [float(c) for c in color]
                if len(color) != 3:
                    raise ValueError
                self.files[name].actor.setColor(color)
            except ValueError:
                show_message("Invalid colour: {}".format(color))
                continue

            # Change opacity
            try:
                self.files[name].actor.setOpacity(float(opacity))
            except ValueError:
                show_message("Invalid opacity: {}".format(opacity))
            
            # transform = vtk.vtkTransform()
            # transform.Translate(1.0, 0.0, 0.0)
            # axes = vtk.vtkAxesActor()
            # #  The axes are positioned with a user transform
            # axes.SetUserTransform(transform)
            self.renWin.renderActors(render)
        self.renWin.addTriad(render, color = [0, 0, 0])
        # print(self.renWin.lim)

    def align(self):
        """
        Numpy style docstring.

        """
        if self.objectsReady(1):
            self.alCont = AlignControls(self.filesDrop, self)
            self.alCont.show()
            self.alCont.centre.clicked.connect(self.centreMesh)
            self.alCont.icp.clicked.connect(self.runICP)
            self.alCont.xrotButton.buttonClicked[QAbstractButton].connect(self.rotatex)
            self.alCont.yrotButton.buttonClicked[QAbstractButton].connect(self.rotatey)
            self.alCont.zrotButton.buttonClicked[QAbstractButton].connect(self.rotatez)
            self.alCont.xtraButton.buttonClicked[QAbstractButton].connect(self.transx)
            self.alCont.ytraButton.buttonClicked[QAbstractButton].connect(self.transy)
            self.alCont.ztraButton.buttonClicked[QAbstractButton].connect(self.transz)
        else:
            show_message("Must be at least 1 object loaded to run align")
    
    def face_Align(self):
        """
        Waits for a point click to occur before calling further functions
        TODO: Create 'Picker controls'? Similar to Alignment controls, but where
        user can enter the name of the point they select - this can allow
        multiple landmark locations to be stored and marked?
        """
        
        if self.objectsReady(2):
            self.face_alCont = Face_AlignControls(self.filesDrop, self)
            self.face_alCont.show()
            # print(dir(self.face_alCont.confirmButton))
            self.face_alCont.startloc.clicked.connect(self.face_start_Point_Pick)
            self.face_alCont.toploc.clicked.connect(self.face_top_Point_Pick)
            self.face_alCont.bottomloc.clicked.connect(self.face_bottom_Point_Pick)
            self.face_alCont.leftloc.clicked.connect(self.face_left_Point_Pick)
            self.face_alCont.rightloc.clicked.connect(self.face_right_Point_Pick)
            self.face_alCont.confirmloc.clicked.connect(self.face_roughICP)
            self.face_alCont.removeloc.clicked.connect(self.face_deloutlyers)
            self.face_alCont.faceICP.clicked.connect(self.face_ICP)
            self.face_alCont.faceregistration.clicked.connect(self.face_registration)
            self.face_alCont.saveface_all.clicked.connect(self.face_save_all)
        else:
            show_message("Must be at least 2 object loaded to run align")
            
    def face_roughICP(self):
        self.face_alCont.consolewidget.append('rough alignment confirmed')
        face = str(self.face_alCont.face.currentText())
        mask = str(self.face_alCont.mask.currentText())
        
        idmask = np.zeros(4)
        idmask[0] = np.argmax(self.files[mask].vert[:,1])
        idmask[1] = np.argmin(self.files[mask].vert[:,1])
        idmask[2] = np.argmax(self.files[mask].vert[:,0])
        idmask[3] = np.argmin(self.files[mask].vert[:,0])
        idmask = idmask.astype(int)
        
        faces = np.array([[0, 2, 3],[1, 2, 3]])
        ampface = dict(zip(['vert', 'faces', 'values'], 
                           [self.files[face].vert[self.face_alCont.face.pnt,:],
                            faces, 
                            self.files[face].values[self.face_alCont.face.pnt]]))
        ampmask = dict(zip(['vert', 'faces', 'values'], 
                           [self.files[mask].vert[idmask,:],
                            faces, 
                            self.files[mask].values[idmask]]))
        self.ampface = AmpObject(ampface, stype='limb')
        self.ampmask = AmpObject(ampmask, stype='limb')
        
        face_rough = ampscanGUI.face_linPoint2Point(self, self.ampface, self.ampmask, self.files[face])
        face_rough.tform = vtk.vtkTransform()
        face_rough.tform.PostMultiply()
        face_rough.addActor()
        face_rough.actor.SetUserTransform(face_rough.tform)
        
        roughName = face + '_rough'
        self.files[roughName] = face_rough
        self.filesDrop.append(roughName)
        self.fileManager.addRow(roughName, self.files[roughName])
        self.fileManager.setTable(mask, [1,0,0], 1, 2)
        self.fileManager.setTable(face, [1,1,1], 1, 0)
        self.fileManager.setTable(roughName, [0,0,1], 1, 2)
        if hasattr(self, 'alCont'):
            self.face_alCont.getNames()
        if hasattr(self, 'regCont'):
            self.face_regCont.getNames()
        
    def face_linPoint2Point(self, ampface, ampmask, face_unal):
        
        faceData = dict(zip(['vert', 'faces', 'values'], 
                         [face_unal.vert, face_unal.faces, face_unal.values]))
        roughData = copy.deepcopy(faceData)
        self.face_rough = AmpObject(roughData, stype='reg')
        
        maxiter = 10
        Rs = np.zeros([3, 3, maxiter+1])
        Ts = np.zeros([3, maxiter+1])
        err = np.zeros([maxiter+1])
        initTransform = np.eye(4)
        Rs[:, :, 0] = initTransform[:3, :3]
        Ts[:, 0] = initTransform[3, :3]
        
        # fC = self.ampface.vert[self.ampface.faces].mean(axis=1)
        # kdTree = spatial.cKDTree(fC)
        self.ampface.rigidTransform(Rs[:, :, 0], Ts[:, 0])
        self.face_rough.rigidTransform(Rs[:, :, 0], Ts[:, 0])
        inlier = 1.0
        inlier = math.ceil(self.ampface.vert.shape[0]*inlier)
        dist = np.sqrt(np.sum((self.ampface.vert - self.ampmask.vert)**2, axis=1))
        idx = np.array([0, 1, 2, 3])
        # Sort by distance
        sort = np.argsort(dist)

        # Keep only those within the inlier fraction
        [dist, idx] = [dist[sort], idx[sort]]
        [dist, idx, sort] = dist[:inlier], idx[:inlier], sort[:inlier]
        err[0] = np.sqrt(dist.mean())
        
        for i in range(maxiter):
            [R,T] = getattr(self, 'face_optDistinit')(self.ampface.vert[sort,:],
                                                      self.ampmask.vert[sort,:])
            Rs[:, :, i+1] = np.dot(R, Rs[:, :, i])
            Ts[:, i+1] = np.dot(R, Ts[:, i]) + T
            self.ampface.rigidTransform(R, T)
            self.face_rough.rigidTransform(R, T)
            dist = np.sqrt(np.sum((self.ampface.vert - self.ampmask.vert)**2,axis=1))
            idx = np.array([0, 1, 2, 3])
            # [dist, idx] = kdTree.query(self.ampface.vert, 1)
            sort = np.argsort(dist)
            [dist, idx] = [dist[sort], idx[sort]]
            [dist, idx, sort] = dist[:inlier], idx[:inlier], sort[:inlier]
            err[i+1] = np.sqrt(dist.mean())
        R = Rs[:, :, -1]
        [U, s, V] = np.linalg.svd(R)
        R = np.dot(U, V)
        # self.files[face].rigidTransform(R,T)
        self.tForm = np.r_[np.c_[R, np.zeros(3)], np.append(Ts[:, -1], 1)[:, None].T]
        self.R = R
        self.T = Ts[:, -1]
        self.rmse = err[-1]
        
        self.face_alCont.consolewidget.append('R : '+str(self.R))
        self.face_alCont.consolewidget.append('T : '+str(self.T))
        self.face_alCont.consolewidget.append('RMSE : '+str(self.rmse))
        
        for i in range(4):
            vtkRenWin.mark(self.renWin, self.ampmask.vert[i,0], 
                            self.ampmask.vert[i,1], self.ampmask.vert[i,2])
            vtkRenWin.mark(self.renWin, self.ampface.vert[i,0], 
                            self.ampface.vert[i,1], self.ampface.vert[i,2])
        
        self.renWin.Render()

        return(self.face_rough)
        
    @staticmethod
    def face_optDistinit(mv, sv):
        
        mCent = mv - mv.mean(axis=0)
        sCent = sv - sv.mean(axis=0)
        C = np.dot(mCent.T, sCent)
        [U,_,V] = np.linalg.svd(C)
        det = np.linalg.det(np.dot(U, V))
        sign = np.eye(3)
        sign[2,2] = np.sign(det)
        R = np.dot(V.T, sign)
        R = np.dot(R, U.T)
        T = sv.mean(axis=0) - np.dot(R, mv.mean(axis=0))
        return (R, T)
    
    def face_deloutlyers(self):
        
        face = str(self.face_alCont.face.currentText())
        mask = str(self.face_alCont.mask.currentText())
        roughName = face + '_rough'
        
        fileNames = []
        for i in range(self.fileManager.n):
            fileNames.append(self.fileManager.getRow(i)[0])
        if roughName in fileNames:
            kdTree = spatial.cKDTree(self.files[mask].vert)
            # faceid = kdTree.query_ball_point(self.files[mask].vert, 20)
            # faceid = faceid[0]
            fC = self.files[roughName].vert[self.files[roughName].faces].mean(axis=1)
            [dist, idx] = kdTree.query(fC,1)
            faceid = []
            for i in range(len(dist)):
                if np.absolute(dist[i]) < 20:
                    faceid.append(i)
                    
            vertid = []
            faces = np.zeros([len(faceid), 3], dtype=int)
            for i in faceid:
                vertid.append(self.files[roughName].faces[i,0])
                vertid.append(self.files[roughName].faces[i,1])
                vertid.append(self.files[roughName].faces[i,2])
            vertid = list(dict.fromkeys(vertid))
            vertices = self.files[roughName].vert[vertid,:]
            
            dictionary = dict()
            for i in range(len(vertices)):
                dictionary[vertid[i]] = i
            for i in range(len(faces)):
                faces[i,0] = dictionary[self.files[roughName].faces[faceid[i],0]]
                faces[i,1] = dictionary[self.files[roughName].faces[faceid[i],1]]
                faces[i,2] = dictionary[self.files[roughName].faces[faceid[i],2]]
            
            roughData = dict(zip(['vert', 'faces', 'values'], 
                              [vertices, 
                               faces, 
                               self.files[roughName].values[vertid]]))
            preppedData = copy.deepcopy(roughData)
            face_prepared = AmpObject(preppedData, stype='reg')
            self.face_alCont.consolewidget.append('# vertices:'+str(face_prepared.vert.shape))
            self.face_alCont.consolewidget.append('# faces:'+str(face_prepared.faces.shape))
    
            face_prepared.tform = vtk.vtkTransform()
            face_prepared.tform.PostMultiply()
            face_prepared.addActor()
            face_prepared.actor.SetUserTransform(face_prepared.tform)
            preparedName = str(self.face_alCont.face.currentText()) + '_prepared'
            self.files[preparedName] = face_prepared
            self.filesDrop.append(preparedName)
            self.fileManager.addRow(preparedName, self.files[preparedName])
            self.fileManager.setTable(mask, [1,0,0], 1, 2)
            self.fileManager.setTable(face, [1,1,1], 1, 0)
            self.fileManager.setTable(roughName, [0,0,1], 1, 0)
            self.fileManager.setTable(preparedName, [0,0,1], 1, 2)
            if hasattr(self, 'alCont'):
                self.face_alCont.getNames()
            if hasattr(self, 'regCont'):
                self.face_regCont.getNames()
            # self.files[alName].faces = self.files[alName].faces[faceid]
        else:
            show_message("Please run the rough align algorithm")
    
    def face_ICP(self):
        
        face = str(self.face_alCont.face.currentText())
        mask = str(self.face_alCont.mask.currentText())
        face_prepared = face + '_prepared'
        
        fileNames = []
        for i in range(self.fileManager.n):
            fileNames.append(self.fileManager.getRow(i)[0])
        if face_prepared in fileNames:
            face_al = align(self.files[face_prepared], self.files[mask],
                            maxiter=10, method='linPoint2Plane')
            # print(dir(face_al))
            self.face_alCont.consolewidget.append('RMSE : '+str(face_al.rmse))
            
            face_al.m.tform = vtk.vtkTransform()
            face_al.m.tform.PostMultiply()
            face_al.m.addActor()
            face_al.m.actor.SetUserTransform(face_al.m.tform)
            
            alName = face+'_al'
            self.files[alName] = face_al.m
            self.filesDrop.append(alName)
            self.fileManager.addRow(alName, self.files[alName])
            self.fileManager.setTable(mask, [1,0,0], 1, 2)
            self.fileManager.setTable(face, [1,1,1], 1, 0)
            self.fileManager.setTable(face_prepared, [0,0,1], 1, 0)
            self.fileManager.setTable(alName, [0,0,1], 1, 2)
            if hasattr(self, 'alCont'):
                self.alCont.getNames()
            if hasattr(self, 'regCont'):
                self.regCont.getNames()
        else:
            show_message("Please run the rough align algorithm")
        
    def face_registration(self):
        
        mask = str(self.face_alCont.mask.currentText())
        face = str(self.face_alCont.face.currentText())
        face_al = str(self.face_alCont.face.currentText())+'_al'
        
        fileNames = []
        for i in range(self.fileManager.n):
            fileNames.append(self.fileManager.getRow(i)[0])
        if face_al in fileNames:
            c1 = [31.0, 73.0, 125.0]
            c3 = [170.0, 75.0, 65.0]
            c2 = [212.0, 221.0, 225.0]
            CMap1 = np.c_[[np.linspace(st, en) for (st, en) in zip(c1, c2)]]
            CMap2 = np.c_[[np.linspace(st, en) for (st, en) in zip(c2, c3)]]
            CMap = np.c_[CMap1[:, :-1], CMap2]
            self.CMapN2P = np.transpose(CMap)/255.0
            self.CMap02P = np.flip(np.transpose(CMap1)/255.0, axis=0)
            
            mask = str(self.face_alCont.mask.currentText())
            face = str(self.face_alCont.face.currentText())
            face_al = str(self.face_alCont.face.currentText())+'_al'
            
            self.fileManager.setTable(mask, [1,0,0], 0.5, 0)
            self.fileManager.setTable(face_al, [0,0,1], 0.5, 0)
            
            face_reg = registration(self.files[mask], self.files[face_al], steps = 5,
                               smooth=1).reg
            
            face_reg.vert = self.files[mask].vert
            face_reg.faces = self.files[mask].faces
            
            #reg.addActor(CMap = self.CMap02P)
            face_reg.addActor(CMap = self.CMapN2P)
            regName = face + '_reg'
            self.files[regName] = face_reg
            self.filesDrop.append(regName)
            self.fileManager.addRow(regName, self.files[regName])
            if hasattr(self, 'alCont'):
                self.alCont.getNames()
            if hasattr(self, 'regCont'):
                self.regCont.getNames()
            #im = []
            face_reg.actor.setScalarRange([-5,5])
            face_reg.actor.setShading(False)
            
                # reg.plotResults(name="distributionofshapevariance.png")
            self.display()  # Reset which objects are displayed
            self.renWin.setScalarBar(face_reg.actor, title = 'Shape difference, mm')
            
            self.face_alCont.consolewidget.append('Run the Registration code between %s and %s' % (mask, face_al))
            
            self.face_output()
        else:
            show_message("Please run the ICP algorithm")
    
    def face_CMapOut(self, amp, colors):
        """
        Colour Map with 4 views (copied Josh's code)
        """
        titles = ['Anterior']
        fig,axes = plt.subplots(ncols=2)
        cmap = clr.ListedColormap(colors, name='Amp')
        norm = clr.Normalize(vmin=-10,vmax=10)
        cb1 = clb.ColorbarBase(axes[-1], cmap=cmap,norm=norm)
        cb1.set_label('Shape deviation / mm')
        for i, ax in enumerate(axes[:-1]):
            im = amp.genIm(size=[3200, 8000],crop=True, az = 0)[0]
            ax.imshow(im)
            ax.set_title(titles[i])
            ax.set_axis_off()
        #plt.colorbar(CMap)
        fig.set_size_inches([12.5, 4])
        mask = str(self.face_alCont.mask.currentText())
        face = str(self.face_alCont.face.currentText())
        if 'small' in mask:
            filename = face+'_small.png'
        if 'medium' in mask:
            filename = face+'_medium.png'
        if 'large' in mask:
            filename = face+'_large.png'
        plt.savefig(os.getcwd()+'\\output\\'+filename, dpi=600)
    
    def face_output(self):
        mask = str(self.face_alCont.mask.currentText())
        face = str(self.face_alCont.face.currentText())
        face_al = face+'_al'
        face_reg = face+'_reg'
        
        fC = self.files[mask].vert[self.files[mask].faces].mean(axis=1)
        kdTree = spatial.cKDTree(fC)
        [dist, idx] = kdTree.query(self.files[face_al].vert, 1)
        dist = np.absolute(dist)
        #general output values
        err = dist.mean()
        std = np.std(dist)
        idmin = np.argmin(self.files[face_reg].values)
        valuemin = np.min(self.files[face_reg].values)
        idmax = np.argmax(self.files[face_reg].values)
        valuemax = np.max(self.files[face_reg].values)
        
        #specific areas
        idmask = np.zeros(4,  dtype=int)
        idmask[0] = np.argmax(self.files[mask].vert[:,1])
        idmask[1] = np.argmin(self.files[mask].vert[:,1])
        idmask[2] = np.argmax(self.files[mask].vert[:,0])
        idmask[3] = np.argmin(self.files[mask].vert[:,0])
        id_0 = self.files[face_reg].values[idmask[0]]
        id_1 = self.files[face_reg].values[idmask[1]]
        id_2 = self.files[face_reg].values[idmask[2]]
        id_3 = self.files[face_reg].values[idmask[3]]
        
        if 'small' in mask: #order is nose bridge, left cheek, left corner, right cheek, right corner, chin
            speccoor = np.array([[0, (-4.1-22)/2, (-13.2-5.1)/2],
                                 [18.2, -33, -22.9],
                                 [29.5, -71.8, -26.7],
                                 [-20.7, -38, -23.3],
                                 [-29.5, -71.8, -26.7],
                                 [0, (-90-105.3)/2, (-16.9-14.4)/2]])
        elif 'medium' in mask:
            speccoor = np.array([[0, (-19.8-2.5)/2, (-14.4-7.9)/2],
                                 [19.9, -34.2, -27],
                                 [34.3, -76.1, -29.9],
                                 [-19.9, -34.2, -27],
                                 [-34.3, -76.1, -29.9],
                                 [0, (-94.6-107.1)/2, (-18.3-14.7)/2]])

        elif 'large' in mask:
            speccoor = np.array([[0, (-0.9-16)/2, (-16.7-11.4)/2],
                                 [18.5, -31.3, -28.3],
                                 [34.2, -73.9, -31.4],
                                 [-18.5, -31.3, -28.3],
                                 [-34.2, -73.9, -31.4],
                                 [0, (-100-117.9)/2, (-22.3-17.8)/2]])
        idspec = []
        distspec = []
        meanspec = np.zeros(speccoor.shape[0])
        stdspec = np.zeros(speccoor.shape[0])
        for i in range(len(speccoor[:,0])):
            idspec.append(kdTree.query_ball_point(speccoor[i,:], 5))
            distspec.append(dist[idspec[i][:]])
            meanspec[i] = np.mean(distspec[i][:])
            stdspec[i] = np.std(distspec[i][:])
        
        #symmetry
        idxleft = np.where(self.files[mask].vert[self.files[mask].faces].mean(axis=1)[:,0] > 0)
        idxright = np.where(self.files[mask].vert[self.files[mask].faces].mean(axis=1)[:,0] < 0)
        distleft = dist[idxleft]
        distright = dist[idxright]
        errleft = distleft.mean()
        stdleft = np.std(distleft)
        errright = distright.mean()
        stdright = np.std(distright)
        
        #write to dict
        self.face_outputfile = {'Total mean (absolute) distance':err,
                                'Total standard deviation':std,
                                'Left mean (absolute) distance':errleft,
                                'Left standard deviation':stdleft,
                                'Right mean (absolute) distance':errright,
                                'Right standard deviation':stdright,
                                'Distance at the bridge of the nose':meanspec[0],
                                'Distance at the chin':meanspec[5],
                                'Distance at the left cheek':meanspec[1],
                                'Distance at the right cheek':meanspec[3],
                                'Distance at the left mouth corner':meanspec[2],
                                'Distance at the right mouth corner':meanspec[4]}
        
    def face_save_all(self):
        face = str(self.face_alCont.face.currentText())
        mask = str(self.face_alCont.mask.currentText())
        alName = face + '_al'
        regName = face + '_reg'
        
        fileNames = []
        for i in range(self.fileManager.n):
            fileNames.append(self.fileManager.getRow(i)[0])
        if (alName in fileNames and
        regName in fileNames):
            
            if 'small' in mask:
                al_output = alName+'_small.stl'
                reg_output = regName+'_small.csv'
                spec_output = face+'_small_spec_output.csv'
            elif 'medium' in mask:
                al_output = alName+'_medium.stl'
                reg_output = regName+'_medium.csv'
                spec_output = face+'_medium_spec_output.csv'
            elif 'large' in mask:
                al_output = alName+'_large.stl'
                reg_output = regName+'_large.csv'
                spec_output = face+'_large_spec_output.csv'
    
            self.files[alName].save(os.getcwd()+'\\output\\'+al_output)
            
            with open(os.getcwd()+'\\output\\'+reg_output, 'w', newline='') as myfile:
                writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                for i in self.files[regName].values:
                    writer.writerow([i])
            
            with open(os.getcwd()+'\\output\\'+spec_output, 'w', newline='') as myfile:
                writer = csv.DictWriter(myfile, fieldnames=['Name', 'Value'])
                for i in self.face_outputfile:
                    writer.writerow({'Name':i, 'Value':self.face_outputfile[i]})
            
            self.face_CMapOut(self.files[regName], colors=self.CMapN2P)
            
            self.face_alCont.consolewidget.append('saved al-file, reg-file and specific output file')
        else:
            show_message("Please run the registration algorithm")
        
    def face_start_Point_Pick(self):
        """
        Waits for a point click to occur before calling further functions
        TODO: Create 'Picker controls'? Similar to Alignment controls, but where
        user can enter the name of the point they select - this can allow
        multiple landmark locations to be stored and marked?
        """
        
        face = str(self.face_alCont.face.currentText())
        # mask = str(self.face_alCont.mask.currentText())
        render = [self.files[face].actor]

        self.renWin.renderActors(render)
        
    def face_top_Point_Pick(self):
        """
        Waits for a point click to occur before calling further functions
        TODO: Create 'Picker controls'? Similar to Alignment controls, but where
        user can enter the name of the point they select - this can allow
        multiple landmark locations to be stored and marked?
        """
        self.face_alCont.consolewidget.append('select point at the bridge of the nose')
        self.face_alCont.lastpressed = 'top'
        self.vtkWidget.iren.AddObserver('RightButtonPressEvent', self.face_loc)
        self.renWin.Render()
        
    def face_bottom_Point_Pick(self):
        """
        Waits for a point click to occur before calling further functions
        TODO: Create 'Picker controls'? Similar to Alignment controls, but where
        user can enter the name of the point they select - this can allow
        multiple landmark locations to be stored and marked?
        """
        self.face_alCont.consolewidget.append('select point below the chin')
        self.face_alCont.lastpressed = 'bottom'
        self.vtkWidget.iren.AddObserver('RightButtonPressEvent', self.face_loc)
        self.renWin.Render()
    
    def face_left_Point_Pick(self):
        """
        Waits for a point click to occur before calling further functions
        TODO: Create 'Picker controls'? Similar to Alignment controls, but where
        user can enter the name of the point they select - this can allow
        multiple landmark locations to be stored and marked?
        """
        self.face_alCont.consolewidget.append('select point on the left cheek')
        self.face_alCont.lastpressed = 'left'
        self.vtkWidget.iren.AddObserver('RightButtonPressEvent', self.face_loc)
        self.renWin.Render()
        
    def face_right_Point_Pick(self):
        """
        Waits for a point click to occur before calling further functions
        TODO: Create 'Picker controls'? Similar to Alignment controls, but where
        user can enter the name of the point they select - this can allow
        multiple landmark locations to be stored and marked?
        """
        self.face_alCont.consolewidget.append('select point on the right cheek')
        self.face_alCont.lastpressed = 'right'
        self.vtkWidget.iren.AddObserver('RightButtonPressEvent', self.face_loc)
        self.renWin.Render()
    
    def face_loc(self, event, x):
        """
        calcs the location of click in GUI (x,y)
        calls function in ampVis.py which converts from GUI coordinates to
        mesh coordinates and marks the point
        """
        idx = {'top':0, 'bottom':1, 'left':2, 'right':3}
        # print(event, x)
        self.vtkWidget.iren.RemoveObservers('RightButtonPressEvent')
        loc = event.GetEventPosition()
        
        # Currently this only allow one pick points, but in the future, more reference points may be needed
        if not hasattr(self.face_alCont.face, 'pnt'):  # Check no points are already picked
            self.face_alCont.face.pnt = [0]*4
        
        x, y = loc
        renderer = self.renWin.rens[0]
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.01)
        picker.Pick(x, y, 0, renderer)
        coor = picker.GetPickPosition()
        vtkRenWin.mark(self.renWin, coor[0], coor[1], coor[2])
        
        self.face_alCont.consolewidget.append(str(picker.GetPointId()))
        # coor = vtkRenWin.Pick_point(self.renWin, loc)
        self.face_alCont.face.pnt[idx[self.face_alCont.lastpressed]] = picker.GetPointId()
        
        self.face_alCont.consolewidget.append('point at '+self.face_alCont.lastpressed+' is set')
        self.face_alCont.lastpressed = ''
    
    def Point_Pick(self):
        """
        Waits for a point click to occur before calling further functions
        TODO: Create 'Picker controls'? Similar to Alignment controls, but where
        user can enter the name of the point they select - this can allow
        multiple landmark locations to be stored and marked?
        """
        self.vtkWidget.iren.AddObserver('RightButtonPressEvent', self.pick_loc)
        self.renWin.Render()
        
    def pick_loc(self, event, x):
        """
        calcs the location of click in GUI (x,y)
        calls function in ampVis.py which converts from GUI coordinates to
        mesh coordinates and marks the point
        """
        #print(event, x)
        self.vtkWidget.iren.RemoveObservers('RightButtonPressEvent')
        loc = event.GetEventPosition()

        # Currently this only allow one pick points, but in the future, more reference points may be needed
        if self.pnt is None:  # Check no points are already picked
            self.pnt = vtkRenWin.Pick_point(self.renWin, loc)
        else:
            show_message("A point is already set as the reference.\n"
                         "Clear the picked points to change reference",
                         message_type="info")
        #vtkRenWin.mark(self.renWin,self.pnt[0],self.pnt[1],self.pnt[2])
        # print(self.pnt)

    def removePick(self):
        """
        delete all marked points and labels
        TODO: be able to delete individual points?
        """
        self.pnt = None
        vtkRenWin.delMarker(self.renWin)

    def rotatex(self, button):
        moving = str(self.alCont.moving.currentText())
        ang = float(button.text())
        idx = [1, 0, 0]
        self.files[moving].rotateAng([ang*i for i in idx], 'deg')
        self.files[moving].tform.RotateX(ang)
        self.renWin.Render()
#        print('rotate x by %.1f' % ang)

    def rotatey(self, button):
        moving = str(self.alCont.moving.currentText())
        ang = float(button.text())
        idx = [0, 1, 0]
        self.files[moving].rotateAng([ang*i for i in idx], 'deg')
        self.files[moving].tform.RotateY(ang)
        self.renWin.Render()
#        print('rotate y by %.1f' % ang)

    def rotatez(self, button):
        moving = str(self.alCont.moving.currentText())
        ang = float(button.text())
        idx = [0, 0, 1]
        self.files[moving].rotateAng([ang*i for i in idx], 'deg')
        self.files[moving].tform.RotateZ(ang)
        self.renWin.Render()

    def transx(self, button):
        moving = str(self.alCont.moving.currentText())
        t = [float(button.text()),0, 0]
        self.files[moving].translate(t)
        self.files[moving].tform.Translate(t)
        self.renWin.Render()
#        print('rotate x by %.1f' % ang)

    def transy(self, button):
        moving = str(self.alCont.moving.currentText())
        t = [0, float(button.text()), 0]
        self.files[moving].translate(t)
        self.files[moving].tform.Translate(t)
        self.renWin.Render()
#        print('rotate y by %.1f' % ang)

    def transz(self, button):
        moving = str(self.alCont.moving.currentText())
        t = [0, 0, float(button.text())]
        self.files[moving].translate(t)
        self.files[moving].tform.Translate(t)
        self.renWin.Render()
#        print('rotate z by %.1f' % ang)
#        self.files[moving].rotateAng(ang, 'deg')

    def centreMesh(self):
        moving = str(self.alCont.moving.currentText())
        c = -1 * self.files[moving].vert.mean(axis=0)
        t = c.tolist()
        self.files[moving].centre()
        self.files[moving].tform.Translate(t)
        self.renWin.Render()

    def runICP(self):
        if self.objectsReady(1):
            static = str(self.alCont.static.currentText())
            moving = str(self.alCont.moving.currentText())
            al = align(self.files[moving], self.files[static],
                       maxiter=10, method='linPoint2Plane').m
            al.tform = vtk.vtkTransform()
            al.tform.PostMultiply()
            al.addActor()
            al.actor.SetUserTransform(al.tform)
            alName = moving + '_al'
            self.files[alName] = al
            self.filesDrop.append(alName)
            self.fileManager.addRow(alName, self.files[alName])
            self.fileManager.setTable(static, [1,0,0], 0.5, 2)
            self.fileManager.setTable(moving, [1,1,1], 1, 0)
            self.fileManager.setTable(alName, [0,0,1], 0.5, 2)
            if hasattr(self, 'alCont'):
                self.alCont.getNames()
            if hasattr(self, 'regCont'):
                self.regCont.getNames()
        else:
            show_message("Must be at least 2 objects loaded to run ICP")

    def runRegistration(self):
        if self.objectsReady(2):
            c1 = [31.0, 73.0, 125.0]
            c3 = [170.0, 75.0, 65.0]
            c2 = [212.0, 221.0, 225.0]
            CMap1 = np.c_[[np.linspace(st, en) for (st, en) in zip(c1, c2)]]
            CMap2 = np.c_[[np.linspace(st, en) for (st, en) in zip(c2, c3)]]
            CMap = np.c_[CMap1[:, :-1], CMap2]
            self.CMapN2P = np.transpose(CMap)/255.0
            self.CMap02P = np.flip(np.transpose(CMap1)/255.0, axis=0)
            baseline = str(self.regCont.baseline.currentText())
            target = str(self.regCont.target.currentText())
            self.fileManager.setTable(baseline, [1,0,0], 0.5, 0)
            self.fileManager.setTable(target, [0,0,1], 0.5, 0)
            reg = registration(self.files[baseline], self.files[target], steps = 5,
                               smooth=1).reg
            #reg.addActor(CMap = self.CMap02P)
            reg.addActor(CMap = self.CMapN2P)
            regName = target + '_reg'
            self.files[regName] = reg
            self.filesDrop.append(regName)
            self.fileManager.addRow(regName, self.files[regName])
            if hasattr(self, 'alCont'):
                self.alCont.getNames()
            if hasattr(self, 'regCont'):
                self.regCont.getNames()
            #im = []
            reg.actor.setScalarRange([-5,5])
            reg.actor.setShading(False)
            analyse.CMapOut(reg, colors=self.CMapN2P)
                # reg.plotResults(name="distributionofshapevariance.png")
            self.display()  # Reset which objects are displayed
            self.renWin.setScalarBar(reg.actor, title = 'Shape difference, mm')

            print('Run the Registration code between %s and %s' % (baseline, target))
        else:
            show_message("Must be at least 2 objects loaded to run registration")

    def register(self):
        """
        Numpy style docstring.

        """
        self.regCont = RegistrationControls(self.filesDrop, self)
        self.regCont.show()
        self.regCont.reg.clicked.connect(self.runRegistration)

    def analyse(self):
        """
        Numpy style docstring.

        """
        #self.RegObj.plot_slices()
        if self.AmpObj != None:  # Check object is loaded
            self.AmpObj.vert[:, 0] *= 2
            self.AmpObj.actor.points.Modified()
            #self.renWin.renderActors([self.AmpObj.actor,])
            #self.AmpObj.vert[0,0] = 1
            #self.AmpObj._v = numpy_support.numpy_to_vtk(self.AmpObj.vert)
        else:
            show_message("Please load object first")

    def measure(self):
        # If no point selected condition move to analyse.py
        if not self.objectsReady(1):
            show_message("Please import an object first")
        elif self.pnt is None:
            show_message("Please select a reference point first using pick")
        else:
            [name, _, color, opacity, display] = self.fileManager.getRow(0)
            output_file_path = analyse.MeasurementsOut(self.files[name], self.pnt)
            if output_file_path == 1:
                show_message("Analysis file cannot be found, ensure you have downloaded the pdf files and placed them into the ampscan/analyse folder")
            # Open Report in webbrowser
            else:
                webbrowser.get().open(output_file_path)  # .get() gets the default browser
    
    def createActions(self):
        """
        Numpy style docstring.

        """
        self.openFile = QAction(QIcon('open.png'), 'Open', self,
                                shortcut='Ctrl+O',
                                triggered=self.chooseOpenFile)
        self.saveFile = QAction(QIcon('open.png'), 'Save', self,
                                shortcut='Ctrl+S',
                                triggered=self.saveFile)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                               triggered=self.close)
        self.align = QAction(QIcon('open.png'), 'Align', self,
                                triggered=self.align)
        self.rect = QAction(QIcon('open.png'), 'Register', self,
                                triggered=self.register)
        self.analyse = QAction(QIcon('open.png'), 'Analyse', self,
                                triggered=self.analyse)
        self.pick = QAction(QIcon('open.png'), 'Right click to select Mid-Patella', self,
                                triggered=self.Point_Pick)
        self.removePick = QAction(QIcon('open.png'), 'Clear all picked points', self,
                                triggered = self.removePick)
        self.Measure = QAction(QIcon('open.png'), 'Generate Measurements File', self,
                                triggered = self.measure)
        self.openObjectManager = QAction(QIcon('open.png'), 'Show Object Manager', self,
                                triggered=self.openAmpObjectManager)
        self.face_Align = QAction(QIcon('open.png'), 'face_Align', self,
                                 triggered=self.face_Align)

    def createMenus(self):
        """
        Numpy style docstring.

        """
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction(self.openFile)
        self.fileMenu.addAction(self.saveFile)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        self.alignMenu = self.menuBar().addMenu("&Align")
        self.alignMenu.addAction(self.align)
        self.regMenu = self.menuBar().addMenu("&Registration")
        self.regMenu.addAction(self.rect)
        self.measureMenu = self.menuBar().addMenu("&Measure")
        self.measureMenu.addAction(self.pick)
        self.measureMenu.addAction(self.removePick)
        self.measureMenu.addAction(self.Measure)
        self.viewMenu = self.menuBar().addMenu("&View")
        self.viewMenu.addAction(self.openObjectManager)
        self.viewMenu = self.menuBar().addMenu("&face_Align")
        self.viewMenu.addAction(self.face_Align)

    def openAmpObjectManager(self):
        self.fileManager.show()

    def objectsReady(self, n):
        """Check there are at least n objects loaded

        """
        return len(self.files) >= n

class fileManager(QMainWindow):
    """
    Controls to manage the displayed 
    
    Example
    -------
    Perhaps an example implementation:

    >>> from GUIs.ampscanGUI import ampscanGUI

    """

    def __init__(self, parent = None):
        super(fileManager, self).__init__(parent)
        self.main = QWidget()
        self.table = QTableWidget()
        self.setCentralWidget(self.main)
        self.layout = QGridLayout()
        self.layout.addWidget(self.table, 0, 0)
        self.main.setLayout(self.layout)
        self.setWindowTitle("AmpObject Manager")
        self.table.setRowCount(0)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(['Name', 'Type', 'Colour', 'Opacity', 'Display'])
        self.n = self.table.rowCount()
        # Set the minimum table size to when it is fully expanded
        self.table.setMinimumWidth(self.table.frameWidth()*2
                                   + self.table.horizontalHeader().length()
                                   + self.table.verticalHeader().width())
        
    def addRow(self, name, amp):
        self.table.insertRow(self.n)
        self.table.setItem(self.n, 0, QTableWidgetItem(name))
        self.table.setItem(self.n, 1, QTableWidgetItem(amp.stype))
        self.table.setItem(self.n, 2, QTableWidgetItem(str(amp.actor.GetProperty().GetColor())))
        self.table.setItem(self.n, 3, QTableWidgetItem(str(amp.actor.GetProperty().GetOpacity())))
        chkBoxItem = QTableWidgetItem()
        chkBoxItem.setTextAlignment(Qt.AlignCenter)
        chkBoxItem.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        chkBoxItem.setCheckState(Qt.Checked)       
        
        self.table.setItem(self.n,4,chkBoxItem)
        self.n = self.table.rowCount()
        
    def getRow(self, i):
        row = []
        for r in range(self.table.columnCount() - 1):
            row.append(self.table.item(i, r).text())
        row.append(self.table.item(i, r+1).checkState())
        return row
    
    def setTable(self, name, color = [1.0, 1.0, 1.0], opacity=1.0, display=2):
        for i in range(self.n):
            if self.table.item(i, 0).text() == name:
                self.table.item(i, 2).setText(str(color))
                self.table.item(i, 3).setText(str(opacity))
                self.table.item(i, 4).setCheckState(display)

class Face_AlignControls(QMainWindow):
    
    def __init__(self, names, parent = None):
        super(Face_AlignControls, self).__init__(parent)
        self.main = QWidget()
        self.names = names
        self.mask = QComboBox()
        self.face = QComboBox()
        self.startloc = QPushButton("start selection")
        self.toploc = QPushButton("top")
        self.bottomloc = QPushButton("bottom")
        self.leftloc = QPushButton("left")
        self.rightloc = QPushButton("right")
        self.confirmloc = QPushButton("rough align")
        self.removeloc = QPushButton("remove region outside bounds")
        self.faceICP = QPushButton('run ICP')
        self.faceregistration = QPushButton('run registration')
        self.saveface_all = QPushButton('save')
        self.consolewidget = QTextEdit()
        self.consolewidget.setReadOnly(True)
        
        self.setCentralWidget(self.main)
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel('Face'), 0, 0)
        self.layout.addWidget(QLabel('Mask'), 1, 0)
        
        self.layout.addWidget(self.face, 0, 1)
        self.layout.addWidget(self.mask, 1, 1)
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.startloc, 0)
        vbox1.addWidget(self.toploc, 1)
        vbox1.addWidget(self.bottomloc, 2)
        vbox1.addWidget(self.leftloc, 3)
        vbox1.addWidget(self.rightloc, 4)
        vbox1.addWidget(self.confirmloc, 5)
        vbox1.addWidget(self.removeloc, 6)
        vbox1.addWidget(self.faceICP, 7)
        vbox1.addWidget(self.faceregistration, 8)
        self.layout.addLayout(vbox1, 2, 0)
        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.consolewidget)
        self.layout.addLayout(vbox2, 2, 1)
        self.layout.addWidget(self.saveface_all, 3, 0, -1, -1)
        
        self.main.setLayout(self.layout)
        self.setWindowTitle("Face Alignment Manager")
        self.getNames()
    
    def getNames(self):
        """
        """
        self.mask.clear()
        self.mask.addItems(self.names)
        self.face.clear()
        self.face.addItems(self.names)
        
class AlignControls(QMainWindow):
    """
    Pop up for controls to align the 
    
    Example
    -------
    Perhaps an example implementation:

    >>> from GUIs.ampscanGUI import ampscanGUI

    """

    def __init__(self, names, parent = None):
        super(AlignControls, self).__init__(parent)
        self.main = QWidget()
        self.names = names
        self.static = QComboBox()
        self.moving = QComboBox()
        self.icp = QPushButton("Run ICP")
        self.centre = QPushButton("Centre")
        self.setCentralWidget(self.main)
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel('Static'), 0, 0)
        self.layout.addWidget(QLabel('Moving'), 1, 0)
        self.layout.addWidget(self.static, 0, 1)
        self.layout.addWidget(self.moving, 1, 1)
        self.layout.addWidget(self.centre, 2, 0, 1, -1)
        self.layout.addWidget(self.icp, 3, 0, 1, -1)
        rots = ['x', 'y', 'z']
        vals = ['-5', '-0.5', '+0.5', '+5']
        for i, r in enumerate(rots):
            setattr(self, r + 'rotBox', QHBoxLayout())
            setattr(self, r + 'rotButton', QButtonGroup())
            lab = QLabel(r + ' rotation')
            getattr(self, r + 'rotBox').addWidget(lab)
            for v in vals:
                button = QPushButton(v)
                getattr(self, r + 'rotBox').addWidget(button)
                getattr(self, r + 'rotButton').addButton(button)
            self.layout.addLayout(getattr(self, r + 'rotBox'), i+4, 0, 1, -1)
        for i, r in enumerate(rots):
            setattr(self, r + 'traBox', QHBoxLayout())
            setattr(self, r + 'traButton', QButtonGroup())
            lab = QLabel(r + ' translation')
            getattr(self, r + 'traBox').addWidget(lab)
            for v in vals:
                button = QPushButton(v)
                getattr(self, r + 'traBox').addWidget(button)
                getattr(self, r + 'traButton').addButton(button)
            self.layout.addLayout(getattr(self, r + 'traBox'), i+7, 0, 1, -1)
        self.main.setLayout(self.layout)
        self.setWindowTitle("Alignment Manager")
        self.getNames()
    
    def getNames(self):
        """
        """
        self.static.clear()
        self.static.addItems(self.names)
        self.moving.clear()
        self.moving.addItems(self.names)

     
class RegistrationControls(QMainWindow):
    """
    Pop up for controls to align the 
    
    Example
    -------
    Perhaps an example implementation:

    >>> from GUIs.ampscanGUI import ampscanGUI

    """

    def __init__(self, names, parent = None):
        super(RegistrationControls, self).__init__(parent)
        self.main = QWidget()
        self.names = names
        self.baseline = QComboBox()
        self.target = QComboBox()
        self.reg = QPushButton("Run Registration")
        # self.tick = QCheckBox("Generate Output File for Comparison?")
        self.setCentralWidget(self.main)
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel('Baseline'), 0, 0)
        self.layout.addWidget(QLabel('Target'), 1, 0)
        self.layout.addWidget(self.baseline, 0, 1)
        self.layout.addWidget(self.target, 1, 1)
        # self.layout.addWidget(self.tick, 2,1)
        self.layout.addWidget(self.reg, 3, 0, 1, -1)
        self.main.setLayout(self.layout)
        self.setWindowTitle("Registration Manager")
        self.getNames()
    
    def getNames(self):
        """
        """
        self.baseline.clear()
        self.baseline.addItems(self.names)
        self.target.clear()
        self.target.addItems(self.names)

class saveControls(QMainWindow):
    """
    Pop up for controls to align the 
    
    Example
    -------
    Perhaps an example implementation:

    >>> from GUIs.ampscanGUI import ampscanGUI

    """

    def __init__(self, names, parent = None):
        super(saveControls, self).__init__(parent)
        self.main = QWidget()
        self.names = names
        self.files = QComboBox()
        self.files.clear()
        self.files.addItems(names)
        self.save = QPushButton("Save file")
        # self.tick = QCheckBox("Generate Output File for Comparison?")
        self.setCentralWidget(self.main)
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel('File'), 0, 0)
        self.layout.addWidget(self.files, 0, 1)
        # self.layout.addWidget(self.tick, 2,1)
        self.layout.addWidget(self.save, 1, 0, 1, -1)
        self.main.setLayout(self.layout)
        self.setWindowTitle("Save file Manager")



def show_message(message, message_type="err", title="An Error Occured..."):
    """
    Parameters
    ----------
    message : string
        The message to be displayed
    message_type : string
        The type of message e.g. "err" or "info"
    title : string
        The title of the dialog window

    Examples
    --------
    >>> show_message("test")
    >>> show_message("test2", "info", "test")

    """
    dialog = QMessageBox()
    dialog.setText(message)
    dialog.setWindowTitle(title)
    icons = {
        "err": QMessageBox.Critical,
        "info": QMessageBox.Information
    }
    dialog.setIcon(icons[message_type])
    dialog.setStandardButtons(QMessageBox.Ok)

    # Makes sure doesn't close until user closes it
    dialog.exec_()

    return dialog


if __name__ == "__main__":
    app = QApplication(sys.argv)
#    mainWin = AlignControls([''])
    mainWin = ampscanGUI()
    mainWin.show()
    sys.exit(app.exec_())