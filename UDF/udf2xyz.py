#!/opt/local/bin/python

# Create xyz trajectory file from output udf 
# trj.xyz   : particle data
# trjGs.xyz : rigid particle COM data
from UDFWrapper import UDFReader
import sys
import os
import string

# atom symbols for xyz file
# symb = string.ascii_uppercase
symb = ["H", "O", "C", "N", "K"]
symb_len = len(symb)

output = UDFReader("output.udf")
output.printDetails()
xyz = open("trj.xyz", "w")
if output.isRigid:
    xyzGs = open("trjGs.xyz", "w")

for t in range(output.numRecords):
    output.jump(t)
    r = output.getParticleData("R")

    xyz.write(str(output.numBeads)+"\n\n")
    for i in range(output.numBeads):
        species = output.chainSpecID[output.beadChainID[i]]
        xyz.write("%3s %10.5f %10.5f %10.5f\n" % \
                  (tuple(symb[species%symb_len]) + tuple(r[i])))

    if output.isRigid:
        rGs = output.getRigidData("R")
        xyzGs.write(str(output.numChains)+"\n\n")
        for i in range(output.numChains):
            species = output.chainSpecID[i]
            xyzGs.write("%3s %10.5f %10.5f %10.5f\n" % \
                        (tuple(symb[species%symb_len]) + tuple(rGs[i])))
xyz.close()
if output.isRigid:
    xyzGs.close()

