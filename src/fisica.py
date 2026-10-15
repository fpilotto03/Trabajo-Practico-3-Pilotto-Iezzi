"""fisica.py  --cargas y condiciones de borde de la presa.

Todo se arma sobre la estructura `mesh` y usa las funciones de contrato
`obtener_puntos_gauss` y `obtener_N_y_detJ`.  Numeracion de GDL:  nodo i -> u = 2 i, v = 2 i + 1
(mesh['nodes']['dof']).  Unidades SI (N, m, Pa); por metro de profundidad (t = 1).

    fuerzas_volumen   : peso propio  F_e = sum_g w_g N' (rho g) detJ t           (clase 8)
    aristas_en_nset   : aristas de elemento cuyos nodos pertenecen a un conjunto de nodos
    fuerzas_presion   : carga de superficie  F_e = int N' (-p n) ds t ,  p = p(x, y)
    presion_hidrostatica : p = rho_w g (H - y)  para y < H (triangular)
    restricciones_base: GDL fijos (empotrada | deslizante)
    resortes_robin    : condicion de Robin en la base  t = -(kn un n + kt ut tau)      (bonus)
"""
import numpy as np
import scipy.sparse as sp

from .elementos import ARISTAS, obtener_puntos_gauss, obtener_N_y_detJ

G = 9.81            # m/s2
RHO_AGUA = 1000.0   # kg/m3


# --------------------------------------------------------------------------------------------
# Fuerzas de volumen (peso propio del hormigon)
# --------------------------------------------------------------------------------------------
def fuerzas_volumen(mesh, properties, g=G, direccion=(0.0, -1.0)):
    """Vector global F (nDofTot,) con el peso propio: b = rho g * direccion (por unidad de volumen)."""
    nodos = mesh['nodes']['position']
    dof = mesh['nodes']['dof']
    F = np.zeros(dof.size)
    direccion = np.asarray(direccion, float)
    for iEle, conn in enumerate(mesh['elements']['connectivity']):
        tipo = mesh['elements']['type'][iEle]
        prop = properties[mesh['elements']['property'][iEle]]
        xy_e = nodos[conn, :]
        b = prop['rho'] * g * direccion                  # fuerza de volumen (N/m3)
        pts, pesos = obtener_puntos_gauss(tipo, 'rigidez')
        fe = np.zeros(2 * len(conn))
        for (xi, eta), w in zip(pts, pesos):
            N, detJ = obtener_N_y_detJ(xy_e, tipo, xi, eta)
            fe[0::2] += w * N * b[0] * detJ * prop.get('t', 1.0)
            fe[1::2] += w * N * b[1] * detJ * prop.get('t', 1.0)
        F[dof[conn, :].reshape(-1)] += fe
    return F


# --------------------------------------------------------------------------------------------
# Aristas de borde y cargas de superficie
# --------------------------------------------------------------------------------------------
def aristas_en_nset(mesh, nodos_set):
    """Lista de (iEle, arista_local) tal que todos los nodos de la arista estan en `nodos_set`.
    arista_local = (a, b) o (a, b, medio): indices locales (ARISTAS del ROL 1)."""
    pertenece = np.zeros(mesh['nodes']['position'].shape[0], bool)
    pertenece[np.asarray(nodos_set, int)] = True
    out = []
    for iEle, conn in enumerate(mesh['elements']['connectivity']):
        for ar in ARISTAS[mesh['elements']['type'][iEle]]:
            if pertenece[conn[list(ar)]].all():
                out.append((iEle, ar))
    return out


def _forma_1d(n, s):
    """Funciones de forma y derivadas 1D sobre la arista, s en [-1, 1].  Orden de nodos
    (inicio, fin[, medio])."""
    if n == 2:
        return np.array([(1 - s) / 2, (1 + s) / 2]), np.array([-.5, .5])
    return (np.array([s * (s - 1) / 2, s * (s + 1) / 2, 1 - s * s]),
            np.array([s - .5, s + .5, -2 * s]))


def _gauss_1d(n=3):
    return np.polynomial.legendre.leggauss(n)


def _geom_arista(xy_a, s):
    """N(s), dN/ds, tangente dx/ds, jacobiano |dx/ds| y normal exterior (arista antihoraria)."""
    N, dN = _forma_1d(len(xy_a), s)
    tang = dN @ xy_a
    ds = np.hypot(*tang)
    normal = np.array([tang[1], -tang[0]]) / ds
    return N, ds, normal, tang / ds


def presion_hidrostatica(H, rho=RHO_AGUA, g=G):
    """p(x, y) = rho g (H - y) si y < H, 0 si no (agua del lado x<0, presion normal al paramento)."""
    return lambda x, y: rho * g * max(H - y, 0.0)


def fuerzas_presion(mesh, properties, aristas, presion):
    """Carga de superficie de presion p (normal, hacia el solido):  traccion = -p n.

    F_e = sum_g w_g N' (-p n) |dx/ds| t  con 3 puntos de Gauss 1D (exacto para p lineal y
    aristas rectas con interpolacion cuadratica).  `presion(x, y)` devuelve p en un punto."""
    nodos = mesh['nodes']['position']
    dof = mesh['nodes']['dof']
    F = np.zeros(dof.size)
    xg, wg = _gauss_1d(3)
    for iEle, ar in aristas:
        conn = mesh['elements']['connectivity'][iEle]
        prop = properties[mesh['elements']['property'][iEle]]
        nd = conn[list(ar)]
        xy_a = nodos[nd, :]
        fe = np.zeros(2 * len(nd))
        for s, w in zip(xg, wg):
            N, ds, normal, _ = _geom_arista(xy_a, s)
            x, y = N @ xy_a
            trac = -presion(x, y) * normal
            fe[0::2] += w * N * trac[0] * ds * prop.get('t', 1.0)
            fe[1::2] += w * N * trac[1] * ds * prop.get('t', 1.0)
        F[dof[nd, :].reshape(-1)] += fe
    return F


# --------------------------------------------------------------------------------------------
# Condiciones de borde
# --------------------------------------------------------------------------------------------
def restricciones_base(mesh, nodos_base, tipo='empotrada'):
    """Vector booleano `fixed` (nDofTot,) con los GDL restringidos en la base.
    'empotrada'  : u = v = 0 en toda la base.
    'deslizante' : v = 0 en toda la base (la presa puede deslizar en x); como queda un modo de
                   cuerpo rigido horizontal, se fija ademas u del nodo del talon (el que esta
                   mas aguas arriba) y el empuje del agua se descarga en ese unico nodo."""
    dof = mesh['nodes']['dof']
    fixed = np.zeros(dof.size, bool)
    nb = np.asarray(nodos_base, int)
    if tipo == 'empotrada':
        fixed[dof[nb, :].reshape(-1)] = True
    elif tipo == 'deslizante':
        fixed[dof[nb, 1]] = True
        fixed[dof[nb[np.argmin(mesh['nodes']['position'][nb, 0])], 0]] = True
    else:
        raise ValueError("tipo debe ser 'empotrada' o 'deslizante'")
    return fixed


def resortes_robin(mesh, properties, aristas, kn, kt, modo='diagonal'):
    """Matriz de rigidez (sparse) de los resortes de la fundacion (Robin) sobre `aristas`.

    La traccion que ejerce el suelo es  t = -(kn (u.n) n + kt (u.tau) tau)   (kn, kt en N/m3).
    Matriz de resortes por arista:  Kr = int N' (kn n n' + kt tau tau') N ds t .
    modo='consistente' : integral exacta (3 puntos de Gauss).
    modo='diagonal'    : "sumando terminos a la diagonal": cada nodo recibe el resorte por su
                         longitud tributaria  w_i = int N_i ds  (lineal 1/2,1/2; cuadratica
                         1/6,1/6,4/6 de la longitud) -> K_ii += w_i (kn n n' + kt tau tau').
    """
    nodos = mesh['nodes']['position']
    dof = mesh['nodes']['dof']
    nT = dof.size
    filas, cols, vals = [], [], []
    xg, wg = _gauss_1d(3)
    for iEle, ar in aristas:
        conn = mesh['elements']['connectivity'][iEle]
        prop = properties[mesh['elements']['property'][iEle]]
        t = prop.get('t', 1.0)
        nd = conn[list(ar)]
        xy_a = nodos[nd, :]
        m = len(nd)
        Ke = np.zeros((2 * m, 2 * m))
        for s, w in zip(xg, wg):
            N, ds, n, tau = _geom_arista(xy_a, s)
            R = kn * np.outer(n, n) + kt * np.outer(tau, tau)
            if modo == 'consistente':
                Nm = np.zeros((2, 2 * m))
                Nm[0, 0::2] = N
                Nm[1, 1::2] = N
                Ke += w * (Nm.T @ R @ Nm) * ds * t
            elif modo == 'diagonal':
                for i in range(m):
                    Ke[2 * i:2 * i + 2, 2 * i:2 * i + 2] += w * N[i] * R * ds * t
            else:
                raise ValueError("modo debe ser 'diagonal' o 'consistente'")
        gd = dof[nd, :].reshape(-1)
        filas.append(np.tile(gd[:, None], (1, 2 * m)).ravel())
        cols.append(np.tile(gd[None, :], (2 * m, 1)).ravel())
        vals.append(Ke.ravel())
    if not vals:
        return sp.csr_matrix((nT, nT))
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))),
                         shape=(nT, nT)).tocsr()
