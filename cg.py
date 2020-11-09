import copy
import numpy as np

def opt1d(pars, direction, energy, gr, func, x):

  h = pars['h']
  D = pars['D']
  H = pars['H']
  G = pars['G']
  K = pars['K']

  l = copy.deepcopy(direction)
  s = np.dot(l,l)
  l /= np.sqrt(s)

  g1 = np.dot(gr, l)

  it = 1
  while abs(g1)>G:

    d = -g1/h;
    if abs(d)>D:
      d = D if d>=0 else -D
    print("  it:%3d   g1:%+.6e   h:%+.6e   d:%+.6e   e:%+.6e"%(it, g1, h, d, energy), flush=True)
    x += l*d
    g2 = copy.deepcopy(g1)
    energy, gr = func(x)
    g1 = np.dot(gr,l)
    h  = (g1-g2)/d
    if h<H:
      h = H
    if it+1==K:
      h = -1.0;
      printf("  no convergence\n")
      break
    it+=1
  return energy, gr, x, h


def conj_grad(pars, pars1d, func, x):
  h_def = pars1d['h']
  MG = pars['MG']
  K  = pars['K']
  N = len(x)
  E, grad = func(x)
  k = 0
  while True:
    dir_prev = np.zeros(N)
    s1 = 1.0;
    for n in range(N*1000):
      mg = max(abs(grad))
      print()
      print("  IT:%3d   g:%+.6e   e:%+.6e"%(k, mg, E), flush=True)
      print("  -------------------------------------------------------------------------------")
      if mg<MG:
        print("converged (k = "+str(k)+")")
        break
      k+=1
      if k>K:
        break
      s2 = np.dot(grad, grad)
      w  = s2/s1
      direction = grad + dir_prev*w
      E, grad, x, pars1d['h'] = opt1d(pars1d, direction, E, grad, func, x)
      if pars1d['h']<0.0:
        # opt1d did not converge
        pars1d['h'] = h_def
      dir_prev = copy.deepcopy(direction)
      s1 = s2;
    else:
      continue
    break
  return E, grad, x

