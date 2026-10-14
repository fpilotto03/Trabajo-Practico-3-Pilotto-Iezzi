"""patch_test.py  -- patch test de los elementos T3, T6, Q4, Q8 y Q9.

Script aislado (no depende del parser ni de la fisica): usa solamente
src/elementos.py y src/ensamble.py.

Patch test (clase 8 y 9)
------------------------
Se arma un parche rectangular [0,L]x[0,H] con elementos IRREGULARES (los nodos interiores estan
desalineados a proposito; en T6/Q8/Q9 tambien los nodos medios interiores, o sea aristas curvas).
Se impone un estado de tension CONSTANTE (sigma_x, sigma_y, tau_xy o una combinacion) de dos
maneras equivalentes:

  Variante 'cargas' (clase 8, "condicion isostatica a tension constante"):
      se aplican en todo el contorno las fuerzas nodales consistentes con la traccion
      t = sigma * n  y solo se restringen los 3 movimientos de cuerpo rigido
      (apoyo fijo en (0,0) y apoyo movil vertical en (L,0)).
      Las fuerzas consistentes sobre una arista recta de longitud Lc con traccion uniforme son
      Lc*t*(1/2, 1/2) en arista lineal y Lc*t*(1/6, 1/6, 4/6) en arista cuadratica (clase 9).

  Variante 'desplazamientos':
      se impone el campo lineal exacto u = eps_x x + gamma y,  v = eps_y y  en TODOS los nodos del
      contorno y se resuelven los nodos interiores sin cargas.

El elemento pasa el patch test si:
  (a) los desplazamientos nodales coinciden con el campo lineal exacto,
  (b) la tension sigma = D B de en todos los puntos de Gauss y en todos los nodos es exactamente la
      impuesta (constante).

Uso:   python patch_test.py          (imprime la tabla y devuelve 0 si todo pasa)
"""
import os
import sys

import numpy as np
import scipy.sparse.linalg as spla

from src.elementos import (ARISTAS, NODOS_NAT, D_de_prop, matriz_B,
                           obtener_puntos_gauss)
from src.ensamble import crear_mesh, ensamblar_K

TOL = 1e-9
TIPOS = ['T3', 'T6', 'Q4', 'Q8', 'Q9']
# Estados de tension constante [sigma_x, sigma_y, tau_xy] (adimensionales respecto de E)
ESTADOS = {
    'sigma_x': np.array([1.0, 0.0, 0.0]),
    'sigma_y': np.array([0.0, 1.0, 0.0]),
    'tau_xy': np.array([0.0, 0.0, 1.0]),
    'combinado': np.array([1.0, 0.5, 0.3]),
}
PROP = {'E': 1000.0, 'nu': 0.3, 't': 1.0}               # material del patch
PLANOS = ('tension_plana', 'deformacion_plana')          # el patch test se verifica en ambos


# --------------------------------------------------------------------------------------------
# Malla irregular del parche
# --------------------------------------------------------------------------------------------
def malla_patch(tipo, L=2.0, H=1.0, nx=3, ny=2, semilla=3, dist=0.25):
    """Devuelve (nodos, conectividad) de un parche rectangular con nodos interiores desalineados."""
    rng = np.random.default_rng(semilla)
    dx, dy = L / nx, H / ny
    nodos = np.array([[i * dx, j * dy] for j in range(ny + 1) for i in range(nx + 1)])
    nid = lambda i, j: j * (nx + 1) + i
    es_tri = tipo in ('T3', 'T6')
    conn = []
    for j in range(ny):
        for i in range(nx):
            n1, n2, n3, n4 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            if not es_tri:
                conn.append([n1, n2, n3, n4])
            elif (i + j) % 2 == 0:                       # dos triangulos antihorarios por celda
                conn += [[n1, n2, n3], [n1, n3, n4]]
            else:
                conn += [[n1, n2, n4], [n2, n3, n4]]
    interior = lambda p: 1e-9 < p[0] < L - 1e-9 and 1e-9 < p[1] < H - 1e-9
    for k in range(len(nodos)):                          # solo los nodos interiores: bordes rectos
        if interior(nodos[k]):
            nodos[k] += dist * np.array([dx, dy]) * rng.uniform(-1, 1, 2)

    if tipo in ('T6', 'Q8', 'Q9'):                       # elevar el orden: nodos medios (y central)
        nodos = list(map(tuple, nodos))
        medios = {}

        def medio(a, b):
            clave = frozenset((a, b))
            if clave not in medios:
                nodos.append(tuple((np.array(nodos[a]) + np.array(nodos[b])) / 2))
                medios[clave] = len(nodos) - 1
            return medios[clave]

        lin = 'T3' if es_tri else 'Q4'
        nueva = []
        extras = []                                      # indices de nodos medios / centrales
        for c in conn:
            ext = list(c) + [medio(c[a[0]], c[a[1]]) for a in ARISTAS[lin]]
            if tipo == 'Q9':
                nodos.append(tuple(np.mean([nodos[k] for k in c], axis=0)))
                ext.append(len(nodos) - 1)
                extras.append(len(nodos) - 1)
            nueva.append(ext)
        extras += list(medios.values())
        conn = nueva
        nodos = np.array(nodos)
        for k in extras:                                 # aristas interiores curvas (nodos medios)
            if interior(nodos[k]):
                nodos[k] += 0.03 * np.array([dx, dy]) * rng.uniform(-1, 1, 2)
    return np.asarray(nodos, float), np.array(conn)


def aristas_de_borde(tipo, conn):
    """Aristas que pertenecen a un solo elemento: lista de tuplas de nodos globales (a, b[, medio])."""
    cuenta = {}
    for c in conn:
        for ed in ARISTAS[tipo]:
            clave = frozenset((c[ed[0]], c[ed[1]]))
            cuenta.setdefault(clave, []).append(tuple(c[list(ed)]))
    return [v[0] for v in cuenta.values() if len(v) == 1]


# --------------------------------------------------------------------------------------------
# Patch test
# --------------------------------------------------------------------------------------------
def correr_patch(tipo, estado, variante='cargas', L=2.0, H=1.0, semilla=3, plano='tension_plana'):
    """Resuelve el parche y devuelve (error_tensiones, error_desplazamientos) relativos."""
    nodos, conn = malla_patch(tipo, L, H, semilla=semilla)
    mesh = crear_mesh(nodos, conn, tipo)
    prop = dict(PROP, estado=plano)                     # tension plana o deformacion plana
    properties = [prop]
    nDofTot = mesh['nodes']['dof'].size
    K = ensamblar_K(mesh, properties)

    sigma0 = ESTADOS[estado]
    D = D_de_prop(prop)
    eps0 = np.linalg.solve(D, sigma0)                   # deformaciones constantes
    # campo exacto compatible con los apoyos isostaticos: u(0,0)=v(0,0)=v(L,0)=0
    u_ex = np.zeros(nDofTot)
    u_ex[0::2] = eps0[0] * nodos[:, 0] + eps0[2] * nodos[:, 1]
    u_ex[1::2] = eps0[1] * nodos[:, 1]

    F = np.zeros(nDofTot)
    fixed = np.zeros(nDofTot, bool)
    Dglob = np.zeros(nDofTot)                           # desplazamientos prescriptos
    borde = aristas_de_borde(tipo, conn)

    if variante == 'cargas':
        for ed in borde:                                # fuerzas consistentes con t = sigma n
            pa, pb = nodos[ed[0]], nodos[ed[1]]
            tang = pb - pa
            Lc = np.linalg.norm(tang)
            n = np.array([tang[1], -tang[0]]) / Lc      # normal exterior (elementos antihorarios)
            trac = np.array([sigma0[0] * n[0] + sigma0[2] * n[1],
                             sigma0[2] * n[0] + sigma0[1] * n[1]])
            pesos = [0.5, 0.5] if len(ed) == 2 else [1 / 6, 1 / 6, 4 / 6]
            for nodo, w in zip(ed, pesos):
                F[2 * nodo:2 * nodo + 2] += w * Lc * prop['t'] * trac
        n0 = int(np.argmin(np.hypot(nodos[:, 0], nodos[:, 1])))
        n1 = int(np.argmin(np.hypot(nodos[:, 0] - L, nodos[:, 1])))
        fixed[[2 * n0, 2 * n0 + 1, 2 * n1 + 1]] = True  # apoyo fijo + apoyo movil: isostatico
    else:
        for nodo in np.unique(np.concatenate([np.array(e) for e in borde])):
            fixed[2 * nodo:2 * nodo + 2] = True
            Dglob[2 * nodo:2 * nodo + 2] = u_ex[2 * nodo:2 * nodo + 2]

    # Particion de la matriz (igual que en el codigo de MATLAB: fixed / free)
    free = ~fixed
    Dn = Dglob.copy()
    rhs = F[free] - K[free][:, fixed] @ Dglob[fixed]
    Dn[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)

    err_u = np.abs(Dn - u_ex).max() / np.abs(u_ex).max()

    # Tensiones: sigma = D B de en los puntos de Gauss y en los nodos de cada elemento
    err_s = 0.0
    for c in conn:
        xy_e = nodos[c, :]
        de = Dn[mesh['nodes']['dof'][c, :].reshape(-1)]
        puntos = np.vstack([obtener_puntos_gauss(tipo)[0], obtener_puntos_gauss(tipo, 'tensiones')[0],
                            NODOS_NAT[tipo]])
        for xi, eta in puntos:
            B, _ = matriz_B(xy_e, tipo, xi, eta)
            err_s = max(err_s, np.abs(D @ (B @ de) - sigma0).max())
    return err_s / np.abs(sigma0).max(), err_u


def main():
    print(f"{'elemento':9s}{'plano':18s}{'variante':16s}{'estado':11s}{'err. tension':>14s}{'err. despl.':>14s}  resultado")
    todo_ok = True
    for tipo in TIPOS:
        for plano in PLANOS:
            for variante in ('cargas', 'desplazamientos'):
                for estado in ESTADOS:
                    es, eu = correr_patch(tipo, estado, variante, plano=plano)
                    ok = es < TOL and eu < TOL
                    todo_ok &= ok
                    print(f"{tipo:9s}{plano:18s}{variante:16s}{estado:11s}{es:14.2e}{eu:14.2e}  {'PASA' if ok else 'FALLA'}")
    print('\nPATCH TEST:', 'TODOS LOS ELEMENTOS PASAN' if todo_ok else 'HAY ELEMENTOS QUE FALLAN')

    try:                                                 # figura opcional de las mallas usadas
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 5, figsize=(15, 2.6))
        for ax, tipo in zip(axs, TIPOS):
            nodos, conn = malla_patch(tipo)
            for c in conn:
                k = 3 if tipo in ('T3', 'T6') else 4
                p = nodos[list(c[:k]) + [c[0]]]
                ax.plot(p[:, 0], p[:, 1], 'k-', lw=0.8)
            ax.plot(nodos[:, 0], nodos[:, 1], 'k.', ms=3)
            ax.set_title(tipo); ax.set_aspect('equal'); ax.axis('off')
        fig.suptitle('Mallas irregulares del patch test')
        os.makedirs('results', exist_ok=True)
        fig.savefig(os.path.join('results', 'patch_test_rol1.png'), dpi=150, bbox_inches='tight')
    except ImportError:
        pass
    return 0 if todo_ok else 1


if __name__ == '__main__':
    sys.exit(main())
