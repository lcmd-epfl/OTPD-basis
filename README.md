# OTPD-basis

This code supports the paper
> A. Fabrizio, K. R. Briling, D. D. Girardier, and C. Corminboeuf,
“Learning on-top: regressing the on-top pair density for real-space visualization of electron correlation”
[`arXiv:2010.07116 [physics.chem-ph]`](https://arxiv.org/abs/2010.07116)

It is written to optimize of basis set
for decomposition of the on-top pair density (OTPD)
onto atom-centered contributions.

## Requirements
* `python >= 3.6`
* `numpy >= 1.16`
* `scipy >= 1.2`
* [`pyscf >= 1.6`](https://github.com/pyscf/pyscf)

## Usage

### 1. Generate the densities to be fitted
```
python otpd.py [-h] [-g grid] molecule basis output
```
Computes the on-top pair density at the Hartree–Fock level
and writes the molecular structure and OTPD values on a grid 
along with grid points and weights.

#### Command-line arguments
* molecule: `.xyz` file with molecular geometry
* basis: AO basis
* output: `.npz` output file
* grid: `pyscf` grid level

#### Examples
```
python otpd.py mol/xyz/H2.xyz   ccpvtz H2.npz
python otpd.py mol/xyz/H2O.xyz  ccpvtz H2O.npz
python otpd.py mol/xyz/H2O2.xyz ccpvtz H2O2.npz
```

### 2. Optimize the basis set for decomposition
```
optimizer.py [-h] [-e element1 [element2 ...]] -b basis [basis ...]
                  --molecules molecule1 [molecule2 ...] [-g gtol]
                  [--method method] [--check]
```
Finds the exponents of the given basis set 
to minimize the OTPD decomposition error.

#### Command-line arguments
* elements: elements basis is optimized for
* basis: basis set to use as an initial guess in `pyscf` format
* molecules: molecular data files (outputs of `otpd.py`)
* method: gradient-based optimization method (e.g. `cg`, `bfgs`)
* gtol: gradient norm tolerance for optimization
* check: if enabled, check gradient and exit

#### Examples
Optimize exponents for Hydrogen on OTPD of H2:
```
python optimizer.py -b initial/H_N0.txt --molecules mol/otpd/H2.ccpvtz.grid3.npz
```
Optimize exponents for Oxygen on OTPD of H2O and H2O2, using optimized basis for Hydrogen:
```
python optimizer.py -e O -b opt/H_N0.bfgs.txt initial/O_N0.txt \
                    --molecules mol/otpd/H2O.ccpvtz.grid3.npz mol/otpd/H2O2.ccpvtz.grid3.npz
```
Compare numerical and analytical derivatives used for optimization:
```
python optimizer.py -e O -b opt/H_N0.bfgs.txt initial/O_N0.txt \
                    --molecules mol/otpd/H2O.ccpvtz.grid3.npz mol/otpd/H2O2.ccpvtz.grid3.npz\
                    --check
```
