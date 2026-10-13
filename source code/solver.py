r"""solver.py  --  resolucion del sistema  K D = F  con condiciones de borde.

Misma logica que MATLAB (TP1):  se particionan los GDL en libres y fijos,

        [Kff Kfc] [Df]   [Ff]
        [Kcf Kcc] [Dc] = [Fc + R]          ->   Df = Kff \ (Ff - Kfc Dc)     R = Kcf Df + Kcc Dc - Fc

(`isZeroRow`: los GDL libres sin rigidez, filas nulas de K, se descartan igual que en el TP1).
"""
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def resolver(K, F, fixed, Dfijo=None, Kextra=None):
    """Devuelve (D, R).

    K      : sparse (nDof x nDof) ensamblada por el ROL 1.
    F      : (nDof,) fuerzas nodales equivalentes (peso propio + presion...).
    fixed  : (nDof,) booleano, True = GDL con desplazamiento prescripto.
    Dfijo  : (nDof,) valores prescriptos (por defecto 0).
    Kextra : sparse opcional que se suma a K (resortes de Robin).
    R      : (nDof,) reacciones; solo distinto de cero en los GDL fijos.  Con resortes de Robin
             la fuerza que el suelo le aplica a la presa es  -Kextra @ D  (ver reacciones_robin).
    """
    nDofTot = F.size
    fixed = np.asarray(fixed, bool)                      # igual que en MATLAB: fixed(dof) = true
    Dfijo = np.zeros(nDofTot) if Dfijo is None else np.asarray(Dfijo, float)
    K = K.tocsr() if Kextra is None else (K + Kextra).tocsr()

    # Para resolver se quitan las filas y columnas bloqueadas (apoyos) y los GDL vacios.
    free = ~fixed
    isZeroRow = np.asarray(abs(K).max(axis=1).todense()).ravel() < 1e-7   # all(abs(K) < 1e-7, 2)
    if isZeroRow[free].any():
        warnings.warn('Hay GDL libres sin rigidez; se los descarta (desplazamiento 0).')
    reducedDof = free & ~isZeroRow

    # Solver: D(reducedDof) = K(reducedDof, reducedDof) \ F(reducedDof)  (con Dfijo = 0 es identico)
    D = np.zeros(nDofTot)
    D[fixed] = Dfijo[fixed]
    rhs = F[reducedDof] - K[reducedDof][:, fixed] @ D[fixed]
    D[reducedDof] = spla.spsolve(K[reducedDof][:, reducedDof].tocsc(), rhs)

    R = np.zeros(nDofTot)                                # reacciones: R = K D - F en los GDL fijos
    R[fixed] = K[fixed] @ D - F[fixed]
    return D, R


def reacciones_robin(Kextra, D):
    """Fuerzas nodales que el suelo elastico (resortes) aplica a la presa:  -Kextra D."""
    return -(Kextra @ D)
