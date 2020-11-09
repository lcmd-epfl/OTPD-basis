#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import scipy.optimize
from pyscf import gto,df,dft
import pyscf.data
from functions import *


def energy(x):
  exponents = np.exp(x)
  newbasis  = exp2basis(exponents, myelements, basis)
  E = 0.0
  for m in moldata:
    E += energy_mol(newbasis, m)
  return E


def gradient(x):
  exponents = np.exp(x)
  newbasis  = exp2basis(exponents, myelements, basis)

  global it

  E = 0.0
  dE_da = np.zeros(nexp)
  for m,name in zip(moldata,args.molecules):
    E_, dE_da_ = gradient_mol(nexp, newbasis, m)
    E     += E_
    dE_da += dE_da_
    print(os.path.basename(name), 'e =', E_, '(', E_/m['self']*100.0, '%)')
  print('it:', it, E, max(abs(dE_da)))
  dE_da = cut_myelements(dE_da, myelements, bf_bounds)

  it+=1
  print(flush=True)

  dE_dx = dE_da * exponents
  return E, dE_dx


def gradient_only(x):
  return gradient(x)[1]


def read_bases(basis_files):
  basis = {}
  for i in basis_files:
    with open(i, "r") as f:
      addbasis = eval(f.read())
    q = list(addbasis.keys())[0]
    if q in basis.keys():
      print('error: several sets for element', q)
      exit()
    basis.update(addbasis)
  return basis


def make_bf_start():
  nbf = []
  for q in elements:
    nbf.append(len(basis[q]))
  bf_bounds = {}
  for i,q in enumerate(elements):
    start = sum(nbf[0:i])
    bf_bounds[q] = [start, start+nbf[i]]
  return bf_bounds


def make_moldata(fname):
  rho_data = np.load(fname)
  molecule = rho_data['atom'   ]
  rho      = rho_data['rho'    ]
  coords   = rho_data['coords' ]
  weights  = rho_data['weights']
  self = np.einsum('p,p,p->',weights,rho,rho)
  mol = gto.M(atom=str(molecule), basis=basis)

  idx     = []
  centers = []
  for iat in range(mol.natm):
    q = mol._atom[iat][0]
    ib0 = bf_bounds[q][0]
    for ib,b in enumerate(mol._basis[q]):
      l = b[0]
      idx     += [ib+ib0] * (2*l+1)
      centers += [iat]    * (2*l+1)
  idx = np.array(idx)

  distances = np.zeros((mol.natm, len(rho)))
  for iat in range(mol.natm):
    center = mol.atom_coord(iat)
    distances[iat] = np.sum((coords - center)**2, axis=1)

  return {
   'mol'       : mol      ,
   'rho'       : rho      ,
   'coords'    : coords   ,
   'weights'   : weights  ,
   'self'      : self     ,
   'idx'       : idx      ,
   'centers'   : centers  ,
   'distances' : distances
  }

###################################################################

parser = argparse.ArgumentParser(description='Make an initial guess')
parser.add_argument('-e', '--elements', metavar='elements',  type=str,   nargs='+',    help='elements for optimization')
parser.add_argument('-b', '--basis',    metavar='basis',     type=str,   nargs='+',    help='initial df bases', required=True)
parser.add_argument('--molecules',      metavar='molecules', type=str,   nargs='+',    help='molecules', required=True)           # cannot use '-m' because pyscf treats it as memory
parser.add_argument('-g', '--gtol',     metavar='gtol',      type=float, default=1e-7, help='tolerance')
parser.add_argument('--method',         metavar='method',    type=str,   default='CG', help='minimization algoritm')
parser.add_argument('--check',          dest='check', default=False, action='store_true')
args = parser.parse_args()


print(args.basis)
basis = read_bases(args.basis)

elements = sorted(basis.keys(), key=pyscf.data.elements.charge)
if args.elements:
  myelements = args.elements
  myelements.sort(key=pyscf.data.elements.charge)
else:
  myelements = elements
print(myelements, '/', elements)

basis_list = [ i for q in elements for i in basis[q]]
angular_momenta = np.array([ i[0]    for i in basis_list ])
exponents       = np.array([ i[1][0] for i in basis_list ])
nexp = len(basis_list)
bf_bounds = make_bf_start()

print(args.molecules)
moldata = []
for fname in args.molecules:
  moldata.append(make_moldata(fname))
print()

for l,a in zip(angular_momenta, exponents):
  print('l =', l, 'a = ', a)
print()


x0 = np.log(exponents)
x1 = cut_myelements(x0, myelements, bf_bounds)
angular_momenta = cut_myelements(angular_momenta, myelements, bf_bounds)

if args.check:
  it = 0
  gr1 = scipy.optimize.approx_fprime(x1, energy, 1e-4)
  gr2 = gradient_only(x1)
  print()
  print('anal')
  print(gr2)
  print('num')
  print(gr1)
  print('diff')
  print(gr1-gr2)
  print('rel diff')
  print((gr1-gr2)/gr1)
  print()
  exit()

print(args.method, 'tol =', args.gtol)
it = 0

xopt = scipy.optimize.minimize(energy, x1, method=args.method, jac=gradient_only, options={ 'gtol':args.gtol,'disp':True}).x

exponents = np.exp(xopt)
newbasis  = exp2basis(exponents, myelements, basis)
printbasis(newbasis, sys.stdout)

