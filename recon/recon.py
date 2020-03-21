import os
import argparse
import sys


def poissonSingle(binloc, infile, outfile, depth=12):
	command_str = binloc + " --in " + infile +" --out " + outfile + " --depth " + str(depth)
	print(command_str)
	os.system(command_str)