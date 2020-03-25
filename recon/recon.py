import os
import argparse
import sys


def poissonSingle(normloc, binloc, trimloc, infile, normfile, recfile, finalfile, confidence, ksearch=5, samplesPerNode=1.0, depth=12, trimD=7, infnorm = False):
    
    if not infnorm:
        command_norm = normloc + " " + infile + " " + "./" + " " + str(ksearch)
        print(command_norm)
        os.system(command_norm)

    command_rec = binloc + " --in " + normfile + " --out " + recfile + " --depth " + str(depth) + " --linearFit --density" + \
        " --samplesPerNode " + str(samplesPerNode) + " --confidence " + str(confidence)
    print(command_rec)
    os.system(command_rec)

    command_trim = trimloc + " --in " + recfile + " --out " + finalfile + " --trim " + str(trimD)
    print(command_trim)
    os.system(command_trim)

normloc = "./../pointcloudToMesh/build/bin/caln"
binloc = "./../PoissonRecon/Bin/Linux/PoissonRecon"
trimloc = "./../PoissonRecon/Bin/Linux/SurfaceTrimmer"
infile = "uni_loc.xyz"
normfile = "uni_l.ply"
recfile = "uni_rec.ply"
finalfile = "uni_final.ply"
trimD = 7
depth = 12
ksearch = 5
samplesPerNode = 3.0
confidence = 2.0
infnorm = True

poissonSingle(normloc, binloc, trimloc, infile, normfile, recfile, finalfile, confidence, ksearch=ksearch, samplesPerNode=samplesPerNode, depth=depth, trimD=trimD, infnorm=infnorm)
