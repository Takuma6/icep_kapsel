#!/opt/local/bin/python
# Wrapper Classes to automatically read KAPSEL's UDF output
from UDFManager import *
import os.path
import math
import numpy
import sys

#########################
# UDFReader Class V3.02
#########################
# 
# -----------
# VARIABLES:
# -----------
#
# uobj           (UDFManager object)
#
# eqType         (constitutive equation)
# dx             (grid spacing)
# shearFlow      (None/"AC"/"DC")
# shearRate 
# shearFreq
#
# nx,ny,nz       (grid points)
# lx,ly,lz       (box dimensions)
# radius         (particle radius)
#
# objType        (spherical_particle, chain, rigid)
# specData       (Particle_spec[]/Chain_spec[]/Rigid_spec[])
# isChain        (chain/rigid = True, spherical_particle = False)
# isRigid        (rigid = True, chain/spherical_particle = False)
# 
# numSpec        (number of species)
# numChainsSpec  (number of chains/rigids per species)
# numBeadsSpec   (number of beads per species)
# 
# nCum           (cumulative sum over particle species)
# nCumChain      (cumulative sum over rigid/chain species)
# nCumBead       (cumulative sum over beads of rigids/chains)
# 
# numBeads       (total number of particle beads)
# numChains      (total number of rigids/chains)
# 
# beadChainID    (rigid/chain particle of a given bead)
# chainSpecID    (species of a given rigid particle/chain)
#
# numRecords     (total number of records)
# timestep       (current frame number)
# t              (current time)
# dt             (simulation timestep)
#
# ----------
# METHODS:
# ----------
# jump(i)       
#   move to i-th record on file 
#   updates timestep and t
#
# getParticleData("X")
#   return numpy array with "Particles[].X" data
# getParticleXData(pid, "X")
#   return numpy array with "Particles[pid].X" data
#
# getRigidData("X")
#   return numpy array with "RigidParticles[].X" data
# getRigidXData(pid, "X")
#   return numpy array with "RigidParticles[pid].X" data
#
# getParticleXSeriesData(a, b, pid, "X")
# return numpy array with "Particles[pid].X" data in time
# interval (a,b)
#
# getRigidXSeriesData(a, b, pid, "X")
# return numpy array with "RigidParticles[pid].X" dat in time
# interval (a,b)
#
# printDetails()
#   print summary of simulation parameters
#
# -----------
# Notes:
# -----------
# - loop over particle beads of species i
#   for j in range(nCum[i], nCum[i+1])
#
# - loop over rigid/chains of species i
#   for j in range(nCumChain[i], nCumChain[i+1])
#
# - loop over beads belonging to rigid/chain i
#   for j in range(nCumBead[i], nCumBead[i+1])
#
# - If isChain == False -> nCum = nCumChain
#   each particle bead is a rigid particle itself
#
# - Which rigid/chain contains bead i ?
#   beadChainID[i]
#
# - What is the species of rigid/chain i ?
#   chainSpecID[i]
#
# - What is the species of bead i ?
#   chainSpecID[beadChainID[i]]
#
# - A rigid is a chain but a chain is not necessarily a rigid
#
######################
# UDFContainer Class
######################
#
# Inherits from UDFReader Class all variables and methods
# Additionaly defines numpy arrays for all particle data
#
# -----------
# VARIABLES:
# -----------
#
# r            (positions of beads)
# r_raw        (positions of beads with no pbc)
# q            (orientation quaternion)
# v            (velocities of beads)
# w            (angular velocity of beads)
# force_h      (hydrodynamic force on beads)
# force_r      (ext/LJ/random force on beads)
# force_s      (slip force on beads)
# torque_h     (hydrodynamic torque on beads)
# torque_r     (ext/LJ/random torque on beads)
# torque_s     (slip torque on beads)
#
# if [isRigid] then the corresponding rigid body variables are defined
#
# rGs          (positions of beads)
# rGs_raw      (positions of beads with no pbc)
# qGs          (orientation quaternion)
# vGs          (velocities of beads)
# wGs          (angular velocity of beads)
# forceGs_h    (hydrodynamic force on beads)
# forceGs_r    (ext/LJ/random force on beads)
# forceGs_s    (slip force on beads)
# torqueGs_h   (hydrodynamic torque on beads)
# torqueGs_r   (ext/LJ/random torque on beads)
# torqueGs_s   (slip torque on beads)
#
# -----------
# METHODS:
# -----------
#
# updateData(i)
#   move to i-th record on file
#   udpate all particle data arrays (r,v,w,...)
# 
class UDFReader:
    'Wrapper Class for UDFManager'
    def __init__(self, fileName):
        self.fileName = fileName
        if os.path.isfile(fileName)==False:
            print( "UDF file does not exist!")
            sys.exit(2)
        self.uobj = UDFManager(fileName)

        ###
        # Check udf version
        if not self.uobj.getEngineVersion() == "v3.02":
            print( "#######################################")
            print( "# Warning   : Engine version mismatch")
            print( "# UDFReader : v3.02")
            print( "# UDF       :",str(self.uobj.getEngineVersion()))
            print( "#######################################")
            print( 

        ###
        # constitutive equation
        self.eqType = self.uobj.get("constitutive_eq.type")
        self.dx = self.uobj.get("constitutive_eq."+self.eqType+".DX")
        if(self.eqType == "Shear_Navier_Stokes" or self.eqType == "Shear_Navier_Stokes_Lees_Edwards"):
            self.shearFlow = self.uobj.get("constitutive_eq."+self.eqType+".External_field.type")
            if(self.shearFlow == "AC"):
                self.shearRate = self.uobj.get("constitutive_eq."+self.eqType+".External_field.AC.Shear_rate")
                self.shearFreq = self.uobj.get("constitutive_eq."+self.eqType+".External_field.AC.Frequency")
            else:
                self.shearRate = self.uobj.get("constitutive_eq."+self.eqType+".External_field.DC.Shear_rate")
                self.shearFreq = float(0)
        else:
            self.shearFlow = None
            self.shearRate = None
            self.shearFreq = None

        ### 
        # Geometry
        self.nx = 2**(self.uobj.get("mesh.NPX"))
        self.ny = 2**(self.uobj.get("mesh.NPY"))
        self.nz = 2**(self.uobj.get("mesh.NPZ"))
        self.lx = self.nx*self.dx
        self.ly = self.ny*self.dx
        self.lz = self.ny*self.dx
        self.radius = (self.uobj.get("A"))*self.dx

        ###
        # time step
        trnx,trny,trnz = numpy.apply_along_axis(int, 0, [[(self.nx+2)/3, (self.ny+2)/3, (self.nz+2)/3]])
        self.dt = numpy.square(self.dx/(2.0*numpy.pi)) / (numpy.square(float(trnx)/float(self.nx)) + \
                                                          numpy.square(float(trny)/float(self.ny)) + \
                                                          numpy.square(float(trnz)/float(self.nz)))


        ###
        # object type
        self.objType = self.uobj.get("object_type.type")
        if self.objType == "spherical_particle":
            self.specData = self.uobj.getArray("object_type.spherical_particle.Particle_spec[]")
            self.isChain = False
            self.isRigid = False
        elif self.objType == "chain":
            self.specData = self.uobj.getArray("object_type.chain.Chain_spec[]")
            self.isChain = True
            self.isRigid = False
        elif self.objType == "rigid":
            self.specData = self.uobj.getArray("object_type.rigid.Rigid_spec[]")
            self.isChain = True
            self.isRigid = True
        self.numSpec = len(self.specData)

        ###
        # species numbers
        self.numChainsSpec = numpy.zeros(self.numSpec,dtype=numpy.int)
        self.numBeadsSpec = numpy.zeros(self.numSpec,dtype=numpy.int)
        if self.isChain:
            for spec in range(self.numSpec):
                self.numBeadsSpec[spec] = int(self.specData[spec][0])
                self.numChainsSpec[spec] = int(self.specData[spec][1])
        else:
            for spec in range(self.numSpec):
                self.numBeadsSpec[spec] = int(1)
                self.numChainsSpec[spec] = int(self.specData[spec][0])

        ###
        # species counters & particle ids
        #
        self.nCum = numpy.zeros(self.numSpec+1, dtype=numpy.int)
        self.nCumChain = numpy.zeros(self.numSpec+1, dtype=numpy.int)
        for spec in range(self.numSpec):
            self.nCum[spec+1] = self.nCum[spec] + int(self.numBeadsSpec[spec]*self.numChainsSpec[spec])
            self.nCumChain[spec+1] = self.nCumChain[spec] + int(self.numChainsSpec[spec])
        self.numBeads = self.nCum[self.numSpec]
        self.numChains = self.nCumChain[self.numSpec]

        dmy = 0
        dmyRigid = 0
        self.nCumBead = numpy.zeros(self.numChains+1, dtype=numpy.int)
        self.beadChainID = numpy.zeros(self.numBeads, dtype=numpy.int)
        self.chainSpecID = numpy.zeros(self.numChains, dtype=numpy.int)
        for spec in range(self.numSpec):
            for rigid in range(self.numChainsSpec[spec]):
                self.chainSpecID[dmyRigid] = spec
                self.nCumBead[dmyRigid+1] = self.nCumBead[dmyRigid] + self.numBeadsSpec[spec]
                for i in range(self.numBeadsSpec[spec]):
                    self.beadChainID[dmy] = dmyRigid
                    dmy = dmy + int(1)
                dmyRigid = dmyRigid + int(1)

        ###
        # Particle trajectory data
        #
        self.numRecords = self.uobj.totalRecord()        
        self.jump(0)

    ### move record
    def jump(self, time):
        res = self.uobj.jump(int(time))
        if(res == int(time)):
            self.timeStep = int(time)
            self.t = self.uobj.get("t")
        return res

    ### get time series data
    def getTimeSeriesData(self, start, stop):
        if(start < 0 or start >= stop or start >= self.numRecords or \
           stop < 0 or stop > self.numRecords):
            return
        np = stop - start
        tData = numpy.zeros(np)
        for i in range(np):
            self.jump(start + i)
            tData[i] = self.t
        return tData

    ### get particle data as numpy arrays
    def getParticleData(self, tag):
        return numpy.asarray(self.uobj.getArray("Particles[]."+tag))
    ### get single particle data as numpy array
    def getParticleXData(self, pid, tag):
        if(pid < 0 or pid >= self.numBeads):
            return
        return numpy.asarray(self.uobj.getArray("Particles["+str(pid)+"]."+tag))

    ### get rigid data as numpy arrays
    def getRigidData(self, tag):
        return numpy.asarray(self.uobj.getArray("RigidParticles[]."+tag))

    ### get single particle data as numpy array
    def getRigidXData(self, pid, tag):
        if(pid < 0 or pid >= self.numChains):
            return
        return numpy.asarray(self.uobj.getArray("RigidParticles["+str(pid)+"]."+tag))

    ### get particle series data
    def getParticleXSeriesData(self, start, stop, pid, tag):
        if(start < 0 or start >= stop or start >= self.numRecords or \
           stop < 0 or stop > self.numRecords or
           pid < 0 or pid >= self.numBeads):
            return

        self.jump(0)
        dp = (self.getParticleXData(pid, tag)).size
        np = stop - start
        pData = numpy.zeros(np*dp).reshape(np, dp)
        for i in range(np):
            self.jump(start + i)
            pData[i] = self.getParticleXData(pid, tag)
        return pData

    ### get rigid particle series data
    def getRigidXSeriesData(self, start, stop, pid, tag):
        if(start < 0 or start >= stop or start >= self.numRecords or \
           stop < 0 or stop > self.numRecords or
           pid < 0 or pid >= self.numChains):
            return

        self.jump(0)
        dp = (self.getRigidXData(pid,tag)).size
        np = stop - start
        pData = numpy.zeros(np*dp).reshape(np, dp)
        for i in range(np):
            self.jump(start + i)
            pData[i] = self.getRigidXData(pid, tag)
        return pData
                                     
    def printDetails(self):
        print( "# UDF file name  : ", self.fileName)
        print( "# Constitutive Eq: ", self.eqType)
        print( "# DX             : ", self.dx)
        print( "# Number Records : ", self.numRecords)
        if self.isRigid:
            print( "# Rigid Particles" )
        elif self.isChain:
            print( "# Flexible Chains" )
        else:
            print( "# Spherical Particles")
        print( "#")
        print( "# Species        : ", self.numSpec)
        print( "# Particle No.   : ", self.numBeads)
        print( "# Rigid No.      : ", self.numChains)
        print( "#")
        print( "# Particle Data:")
        print( "#\t\tId\tRigidId\tSpecies")
        for i in range(self.numBeads):
            print( ("#\t%10i %10i %10i") % (i, self.beadChainID[i], self.chainSpecID[self.beadChainID[i]]))
        print( "#")
        print( "# Rigid Data :")
        print( "#\t\tId\t Start\t End\t Beads\t Species")
        for i in range(self.numChains):
            print( ("#\t%8i %8i %8i %8i %8i") % \
                (i, self.nCumBead[i], self.nCumBead[i+1] -1, \
                 self.nCumBead[i+1]-self.nCumBead[i], self.chainSpecID[i]))

class UDFContainer(UDFReader):
    'Container Class for UDF particle data'
    def __init__(self, fileName):
        UDFReader.__init__(self, fileName)
        self.r = numpy.zeros(self.numBeads*3, dtype=numpy.float64).reshape(self.numBeads, 3)
        self.r_raw = numpy.zeros(self.numBeads*3, dtype=numpy.float64).reshape(self.numBeads, 3)
        self.q = numpy.zeros(self.numBeads*4, dtype=numpy.float64).reshape(self.numBeads, 4)
        
        self.v = numpy.zeros(self.numBeads*3, dtype=numpy.float64).reshape(self.numBeads, 3)
        self.w = numpy.zeros(self.numBeads*3, dtype=numpy.float64).reshape(self.numBeads, 3)
        
        self.force_h = numpy.zeros(self.numBeads*3, dtype=numpy.float64).reshape(self.numBeads, 3)
        self.force_r = numpy.zeros(self.numBeads*3, dtype=numpy.float64).reshape(self.numBeads, 3)
        self.force_s = numpy.zeros(self.numBeads*3, dtype=numpy.float64).reshape(self.numBeads, 3)
        
        self.torque_h = numpy.zeros(self.numBeads*3, dtype=numpy.float64).reshape(self.numBeads, 3)
        self.torque_r = numpy.zeros(self.numBeads*3, dtype=numpy.float64).reshape(self.numBeads, 3)
        self.torque_s = numpy.zeros(self.numBeads*3, dtype=numpy.float64).reshape(self.numBeads,3)
        
        if(self.isRigid):
            self.rGs = numpy.zeros(self.numChains*3, dtype=numpy.float64).reshape(self.numChains, 3)
            self.rGs_raw = numpy.zeros(self.numChains*3, dtype=numpy.float64).reshape(self.numChains, 3)
            self.qGs = numpy.zeros(self.numChains*4, dtype=numpy.float64).reshape(self.numChains, 4)
            
            self.vGs = numpy.zeros(self.numChains*3, dtype=numpy.float64).reshape(self.numChains, 3)
            self.wGs = numpy.zeros(self.numChains*3, dtype=numpy.float64).reshape(self.numChains, 3)
            
            self.forceGs_h = numpy.zeros(self.numChains*3, dtype=numpy.float64).reshape(self.numChains, 3)
            self.forceGs_r = numpy.zeros(self.numChains*3, dtype=numpy.float64).reshape(self.numChains, 3)
            self.forceGs_s = numpy.zeros(self.numChains*3, dtype=numpy.float64).reshape(self.numChains, 3)
            
            self.torqueGs_h = numpy.zeros(self.numChains*3, dtype=numpy.float64).reshape(self.numChains, 3)
            self.torqueGs_r = numpy.zeros(self.numChains*3, dtype=numpy.float64).reshape(self.numChains, 3)
            self.torqueGs_s = numpy.zeros(self.numChains*3, dtype=numpy.float64).reshape(self.numChains, 3)
        else:
            self.rGs = None
            self.rGs_raw = None
            self.qGs = None
            
            self.vGs = None
            self.wGs = None
            
            self.forceGs_h = None
            self.forceGs_r = None
            self.forceGs_s = None
            
            self.torqueGs_h = None
            self.torqueGs_r = None
            self.torqueGs_s = None

    def updateData(self, time):
        if self.jump(time) == time:
            self.r = self.getParticleData("R")
            self.r_raw = self.getParticleData("R_raw")
            self.q = self.getParticleData("q")

            self.v = self.getParticleData("v")
            self.w = self.getParticleData("omega")

            self.force_h = self.getParticleData("f_hydro")
            self.force_r = self.getParticleData("f_r")
            self.force_s = self.getParticleData("f_slip")

            self.torque_h = self.getParticleData("torque_hydro")
            self.torque_r = self.getParticleData("torque_r")
            self.torque_s = self.getParticleData("torque_slip")
            if(self.isRigid):
                self.rGs = self.getRigidData("R")
                self.rGs_raw = self.getRigidData("R_raw")
                self.qGs = self.getRigidData("q")
                
                self.vGs = self.getRigidData("v")
                self.wGs = self.getRigidData("omega")

                self.forceGs_h = self.getRigidData("f_hydro")
                self.forceGs_r = self.getRigidData("f_r")
                self.forceGs_r = self.getRigidData("f_slip")

                self.torque_h = self.getRigidData("torque_hydro")
                self.torque_r = self.getRigidData("torque_r")
                self.torque_s = self.getRigidData("torque_s")
        else:
            print( "Time step out of bounds\n")
                    
