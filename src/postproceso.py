"""postproceso.py  --  tensiones, tensiones principales, deformada y momentos de la presa.

    tensiones_en_gauss        sigma = D B d_e en los puntos de Gauss (muestreo de la clase 9)
    tensiones_nodales         extrapolacion de los puntos de Gauss a los nodos + promedio nodal
    principales               sigma1, sigma2 y angulo (circulo de Mohr)
    momentos_vuelco           momento estabilizante / desestabilizante analitico y con FEM
    graficar_*                deformada con factor de escala y mapas de color
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from .elementos import (NODOS_NAT, ES_TRIANGULO, obtener_puntos_gauss, obtener_N_y_detJ,
                        obtener_B_y_D)
from .fisica import G, RHO_AGUA


# --------------------------------------------------------------------------------------------
# Tensiones
# --------------------------------------------------------------------------------------------
def tensiones_en_gauss(mesh, properties, D, para='tensiones'):
    """Lista (por elemento) de (puntos_nat (ng,2), sigma (ng,3)) con sigma = [sx, sy, txy]."""
    nodos = mesh['nodes']['position']
    dof = mesh['nodes']['dof']
    out = []
    for iEle, conn in enumerate(mesh['elements']['connectivity']):
        tipo = mesh['elements']['type'][iEle]
        prop = properties[mesh['elements']['property'][iEle]]
        xy_e = nodos[conn, :]
        de = D[dof[conn, :].reshape(-1)]
        pts, _ = obtener_puntos_gauss(tipo, para)
        sig = np.zeros((len(pts), 3))
        for g, (xi, eta) in enumerate(pts):
            B, Dm = obtener_B_y_D(xy_e, prop, tipo, xi, eta)
            sig[g] = Dm @ (B @ de)
        out.append((pts, sig))
    return out


def _base_extrapolacion(tipo, pts):
    """Polinomio de ajuste: Q -> bilineal (1, xi, eta, xi eta) con 4 puntos; T6 -> lineal con
    3 puntos; T3 -> constante."""
    pts = np.atleast_2d(pts)
    xi, eta = pts[:, 0], pts[:, 1]
    if tipo == 'T3':
        return np.ones((len(pts), 1))
    if tipo == 'T6':
        return np.column_stack([np.ones(len(pts)), xi, eta])
    return np.column_stack([np.ones(len(pts)), xi, eta, xi * eta])


def tensiones_nodales(mesh, properties, D):
    """Tensiones [sx, sy, txy] en cada nodo (nNodos, 3).

    Para cada elemento se ajusta el polinomio a los valores en los puntos de Gauss y se evalua
    en las coordenadas naturales de sus nodos (clase 9); luego se promedian los valores que
    aportan los elementos que comparten el nodo."""
    nN = mesh['nodes']['position'].shape[0]
    suma = np.zeros((nN, 3))
    cuenta = np.zeros(nN)
    muestras = tensiones_en_gauss(mesh, properties, D, 'tensiones')
    for iEle, conn in enumerate(mesh['elements']['connectivity']):
        tipo = mesh['elements']['type'][iEle]
        pts, sig = muestras[iEle]
        A = _base_extrapolacion(tipo, pts)
        coef = np.linalg.solve(A, sig)                    # (nb, 3)
        valores = _base_extrapolacion(tipo, NODOS_NAT[tipo]) @ coef
        suma[conn] += valores
        cuenta[conn] += 1
    return suma / cuenta[:, None]


def principales(sig):
    """sigma1 >= sigma2 y angulo de sigma1 (rad) a partir de [sx, sy, txy] (n,3) o (3,)."""
    sig = np.atleast_2d(sig)
    sx, sy, txy = sig[:, 0], sig[:, 1], sig[:, 2]
    c = (sx + sy) / 2
    R = np.hypot((sx - sy) / 2, txy)
    return c + R, c - R, 0.5 * np.arctan2(2 * txy, sx - sy)


def von_mises_plano(sig):
    sig = np.atleast_2d(sig)
    sx, sy, txy = sig[:, 0], sig[:, 1], sig[:, 2]
    return np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * txy ** 2)


# --------------------------------------------------------------------------------------------
# Geometria de la seccion y momentos de vuelco respecto del pie
# --------------------------------------------------------------------------------------------
def area_y_centroide(mesh, properties):
    """Area, centroide y masa por metro de la malla, integrando con Gauss: sum w detJ."""
    nodos = mesh['nodes']['position']
    A, Sx, Sy = 0.0, 0.0, 0.0
    for iEle, conn in enumerate(mesh['elements']['connectivity']):
        tipo = mesh['elements']['type'][iEle]
        xy_e = nodos[conn, :]
        pts, pesos = obtener_puntos_gauss(tipo, 'rigidez')
        for (xi, eta), w in zip(pts, pesos):
            N, detJ = obtener_N_y_detJ(xy_e, tipo, xi, eta)
            x, y = N @ xy_e
            A += w * detJ
            Sx += w * detJ * x
            Sy += w * detJ * y
    return A, np.array([Sx / A, Sy / A])


def momentos_vuelco(mesh, properties, nodos_base, R, H, rho_c=None, rho_w=RHO_AGUA, g=G,
                    R_extra=None, nodos_aguas_arriba=None):
    """Momentos respecto del pie (por metro de profundidad) [N m/m].

    Analitico:   M_est = rho_c g A (x_pie - x_G)      (peso propio; + peso del agua si mu>0)
                 M_des = rho_w g H^3 / 6              (empuje hidrostatico, H/3 sobre la base)
                 FS    = M_est / M_des
    FEM:         M_R = sum_i [ (x_i - x_pie) Ry_i - y_i Rx_i ]   (antihorario +) sobre los nodos
                 de la base: es la reaccion del apoyo, que debe equilibrar al par aplicado, por
                 eso  M_R = -(M_est - M_des)  (si las cargas resultan en equilibrio).
    Se devuelve tambien el punto de aplicacion de la reaccion vertical (regla del tercio medio).
    `R` es el vector de reacciones (D empotrada) o, con Robin, las fuerzas del suelo sobre la presa
    con signo de reaccion (ver reacciones_robin); `R_extra` se suma a R si se da.
    """
    nodos = mesh['nodes']['position']
    dof = mesh['nodes']['dof']
    if rho_c is None:
        rho_c = properties[0]['rho']
    A, xyG = area_y_centroide(mesh, properties)
    x_pie = nodos[nodos_base, 0].max()
    x_talon = nodos[nodos_base, 0].min()
    M_hor = rho_c * g * A * (x_pie - xyG[0])             # peso propio del hormigon
    M_des = rho_w * g * H ** 3 / 6
    # Si el paramento aguas arriba esta inclinado (x_up = a + mu y), el agua que apoya sobre el
    # paramento aporta una carga vertical estabilizante  Wa = rho_w g mu H^2 / 2  aplicada a
    # x_talon + mu H / 3  (centroide del triangulo de agua).
    nup = np.asarray(nodos_aguas_arriba, int) if nodos_aguas_arriba is not None else None
    mu = 0.0 if nup is None else float(np.polyfit(nodos[nup, 1], nodos[nup, 0], 1)[0])
    Wa = rho_w * g * mu * H ** 2 / 2
    x_talon0 = nodos[nodos_base, 0].min()
    M_agua_v = Wa * (x_pie - (x_talon0 + mu * H / 3))
    M_est = M_hor + M_agua_v
    Rt = R if R_extra is None else R + R_extra
    nb = np.asarray(nodos_base, int)
    Rx, Ry = Rt[dof[nb, 0]], Rt[dof[nb, 1]]
    x, y = nodos[nb, 0], nodos[nb, 1]
    M_R = np.sum((x - x_pie) * Ry - y * Rx)
    x_res = np.sum(x * Ry) / np.sum(Ry)                   # punto de paso de la reaccion vertical
    B = x_pie - x_talon
    return {'A': A, 'xG': xyG[0], 'yG': xyG[1], 'peso': rho_c * g * A,
            'empuje': rho_w * g * H ** 2 / 2,
            'M_est': M_est, 'M_est_hormigon': M_hor, 'M_agua_v': M_agua_v, 'mu': mu,
            'M_des': M_des, 'FS': M_est / M_des, 'FS_sin_agua': M_hor / M_des,
            'M_R': M_R, 'M_neto': M_est - M_des, 'Rx_tot': Rx.sum(), 'Ry_tot': Ry.sum(),
            'x_pie': x_pie, 'x_talon': x_talon, 'B': B, 'x_res': x_res,
            'excentricidad': (x_talon + x_pie) / 2 - x_res,
            'tercio_medio': bool(x_talon + B / 3 <= x_res <= x_talon + 2 * B / 3)}


# --------------------------------------------------------------------------------------------
# Graficos
# --------------------------------------------------------------------------------------------
def _triangulos_lineales(mesh):
    """Triangulos de esquinas (para colorear) de cada elemento."""
    tri = []
    for conn, tipo in zip(mesh['elements']['connectivity'], mesh['elements']['type']):
        if ES_TRIANGULO[tipo]:
            tri.append(conn[:3])
        else:
            tri += [conn[[0, 1, 2]], conn[[0, 2, 3]]]
    return np.array(tri)


def _aristas_poligonales(mesh, xy):
    """Segmentos de borde de elemento (pasando por el nodo medio en los cuadraticos)."""
    from .elementos import ARISTAS
    seg = set()
    out = []
    for conn, tipo in zip(mesh['elements']['connectivity'], mesh['elements']['type']):
        for ar in ARISTAS[tipo]:
            nd = conn[[ar[0], ar[2], ar[1]]] if len(ar) == 3 else conn[list(ar)]
            clave = frozenset((conn[ar[0]], conn[ar[1]]))
            if clave in seg:
                continue
            seg.add(clave)
            out.append(xy[nd])
    return out


def desplazadas(mesh, D, escala):
    return mesh['nodes']['position'] + escala * D[mesh['nodes']['dof']]


def graficar_deformada(mesh, D, escala, ruta=None, titulo=None, campo=None, etiqueta='',
                       cmap='viridis', ax=None, vmin=None, vmax=None):
    """Malla original (gris) y deformada (xdef = x + escala * u), opcionalmente coloreada con un
    campo nodal `campo` (nNodos,).  Devuelve (fig, ax)."""
    xy0 = mesh['nodes']['position']
    xy = desplazadas(mesh, D, escala)
    propia = ax is None
    if propia:
        fig, ax = plt.subplots(figsize=(8, 6.2))
    else:
        fig = ax.figure
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(_aristas_poligonales(mesh, xy0), colors='0.7',
                                     linewidths=0.5, linestyles='--'))
    if campo is not None:
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1], _triangulos_lineales(mesh))
        cs = ax.tripcolor(tri, campo, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
        cb = fig.colorbar(cs, ax=ax, shrink=0.8)
        cb.set_label(etiqueta)
    ax.add_collection(LineCollection(_aristas_poligonales(mesh, xy), colors='k', linewidths=0.6))
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(titulo or f'Configuracion deformada (factor de escala {escala:g})')
    if propia and ruta:
        fig.savefig(ruta, dpi=140, bbox_inches='tight')
        plt.close(fig)
    return fig, ax


def graficar_sigma1(mesh, properties, D, escala, ruta=None, titulo=None):
    """sigma1 (tension principal maxima) sobre la configuracion deformada, en MPa."""
    sig = tensiones_nodales(mesh, properties, D)
    s1 = principales(sig)[0] / 1e6
    return graficar_deformada(mesh, D, escala, ruta, titulo or f'$\\sigma_1$ [MPa] sobre la deformada '
                              f'(x{escala:g})', campo=s1, etiqueta='sigma_1 [MPa]', cmap='coolwarm')


def tension_en_base(mesh, properties, D, nodos_base):
    """x y sigma_y, tau_xy, sigma_1 [Pa] sobre los nodos de la base ordenados por x."""
    sig = tensiones_nodales(mesh, properties, D)
    nb = np.asarray(nodos_base, int)
    nb = nb[np.argsort(mesh['nodes']['position'][nb, 0])]
    return mesh['nodes']['position'][nb, 0], sig[nb, 1], sig[nb, 2], principales(sig[nb])[0]
