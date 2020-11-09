#!/usr/bin/env python3

import argparse
import numpy as np
from pyscf import gto,scf,dft

density = 2

def readmol(fname, basis, charge=0, spin=0, ignore=False):

  def readxyz(fname):
    with open(fname, "r") as f:
      xyz = f.readlines()
    return "".join(xyz[2:])

  def makemol(xyz, basis, charge=0, spin=0):
    mol = gto.Mole()
    mol.atom = xyz
    mol.charge = charge
    mol.basis = basis
    mol.build()
    return mol

  xyz = readxyz(fname)
  if not ignore:
    mol = makemol(xyz, basis, charge, spin)
  else:
    try:
      mol = makemol(xyz, basis)
    except:
      mol = makemol(xyz, basis, -1)
  return mol

################################################################################

parser = argparse.ArgumentParser(description='Generate a density to be fitted')
parser.add_argument('molecule',     metavar='molecule', type=str, help='xyz file')
parser.add_argument('basis',        metavar='basis',    type=str, help='ao basis')
parser.add_argument('output',       metavar='output',   type=str, help='output file')
parser.add_argument('-g', '--grid', metavar='grid',     type=int, help='grid level', default=3)
args = parser.parse_args()

mol = readmol(args.molecule, args.basis)

mf = scf.RHF(mol)
mf.run()
print("Convergence: ", mf.converged)
print("Energy: ", mf.e_tot)

grid = dft.gen_grid.Grids(mol)
grid.level = args.grid
grid.build()

dm = mf.make_rdm1()
ao = dft.numint.eval_ao(mol, grid.coords)
rho1 = np.einsum('pq,ip,iq->i', dm, ao, ao)

if density == 2:
  rho = rho1*rho1*0.5 # gives OTPD*2
elif density == 1:
  rho = rho1

np.savez(args.output, atom=mol.atom, rho=rho, coords=grid.coords, weights=grid.weights)

