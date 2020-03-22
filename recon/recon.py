import os
import argparse
import sys


def poissonSingle(normloc, binloc, trimloc, infile, normfile, recfile, finalfile, ksearch=5, samplesPerNode=1.0, depth=12, trimD=7):
    

    command_norm = normloc + " " + infile + " " + "./" + " " + str(ksearch)
    print(command_norm)
    os.system(command_norm)

    command_rec= binloc + " --in " + normfile + " --out " + recfile + " --depth " + str(depth) + " --density" + \
        " --samplesPerNode " + str(samplesPerNode)
    print(command_rec)
    os.system(command_rec)

    command_trim = trimloc + " --in " + recfile + " --out " + finalfile + " --trim " + str(trimD)
    print(command_trim)
    os.system(command_trim)

normloc = "./../pointcloudToMesh/build/bin/caln"
binloc = "./../PoissonRecon/Bin/Linux/PoissonRecon"
trimloc = "./../PoissonRecon/Bin/Linux/SurfaceTrimmer"
infile = "gt_pc.xyz"
normfile = "gt_pc.ply"
recfile = "chair_den1.ply"
finalfile = "chair_final1.ply"
trimD = 7
depth = 12
ksearch = 5
samplesPerNode = 1.0

poissonSingle(normloc, binloc, trimloc, infile, normfile, recfile, finalfile, ksearch=ksearch, samplesPerNode=samplesPerNode, depth=depth, trimD=trimD)
