"""geometria.py  -- geometria de la presa de gravedad y generador de mallas.

Perfil (por metro de profundidad, estado plano):
    coronamiento de ancho c a la cota Hd (altura total)
    paramento aguas arriba  x_up(y)   = -mu (Hd - y)        (mu = 0: vertical)
    paramento aguas abajo   x_down(y) =  c + md (Hd - y)
    base de ancho B = c + (mu + md) Hd ; talon (heel) en x_up(0), pie (toe) en x_down(0)
    nivel del agua H = Hd - d  (d = borde libre).  El agua esta del lado x < 0.
"""
from dataclasses import dataclass

import numpy as np

from .elementos import ARISTAS


@dataclass
class Presa:
    Hd: float = 30.0      # altura total
    c: float = 3.0        # cresta
    d: float = 2.0        # borde libre
    B: float = 18.0       # ancho de la base
    mu: float = 0.0       # pendiente del paramento aguas arriba (horizontal/vertical)

    @property
    def md(self):         # pendiente aguas abajo (resulta de B, c y mu)
        return (self.B - self.c) / self.Hd - self.mu

    @property
    def H(self):          # altura del agua
        return self.Hd - self.d

    @property
    def x_talon(self):
        return -self.mu * self.Hd

    @property
    def x_pie(self):
        return self.c + self.md * self.Hd

    def x_up(self, y):
        return -self.mu * (self.Hd - y)

    def x_down(self, y):
        return self.c + self.md * (self.Hd - y)

    def poligono(self):
        return np.array([[self.x_talon, 0], [self.x_pie, 0], [self.c, self.Hd], [0, self.Hd]])

    def area_centroide(self):
        """Area y centroide por la formula del poligono (equivale a sumar el rectangulo y el
        triangulo de la seccion trapezoidal)."""
        p = self.poligono()
        x, y = p[:, 0], p[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        A = cr.sum() / 2
        return A, np.array([((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)])


def generar_malla_presa(presa, tipo, nx=8, ny=30):
    """Malla estructurada de la presa.  Devuelve (nodos, conectividad, nsets).

    Se arma primero la malla lineal (Q4 o T3) por filas horizontales y luego se eleva el orden
    agregando los nodos medios (y el central en Q9).  Se fuerza una fila de nodos a la cota del
    agua para que la presion triangular termine justo en un nodo.
    """
    ys = np.unique(np.round(np.concatenate([np.linspace(0, presa.Hd, ny + 1), [presa.H]]), 9))
    nodos = np.array([[presa.x_up(y) + (presa.x_down(y) - presa.x_up(y)) * i / nx, y]
                      for y in ys for i in range(nx + 1)])
    nid = lambda i, j: j * (nx + 1) + i
    es_tri = tipo in ('T3', 'T6')
    conn = []
    for j in range(len(ys) - 1):
        for i in range(nx):
            n1, n2, n3, n4 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            if not es_tri:
                conn.append([n1, n2, n3, n4])
            elif (i + j) % 2 == 0:
                conn += [[n1, n2, n3], [n1, n3, n4]]
            else:
                conn += [[n1, n2, n4], [n2, n3, n4]]
    if tipo in ('T6', 'Q8', 'Q9'):
        lin = 'T3' if es_tri else 'Q4'
        nodos = list(map(tuple, nodos))
        medios, nueva = {}, []
        for c in conn:
            ext = list(c)
            for a in ARISTAS[lin]:
                clave = frozenset((c[a[0]], c[a[1]]))
                if clave not in medios:
                    nodos.append(tuple((np.array(nodos[c[a[0]]]) + np.array(nodos[c[a[1]]])) / 2))
                    medios[clave] = len(nodos) - 1
                ext.append(medios[clave])
            if tipo == 'Q9':
                nodos.append(tuple(np.mean([nodos[k] for k in c], axis=0)))
                ext.append(len(nodos) - 1)
            nueva.append(ext)
        conn, nodos = nueva, np.array(nodos)
    tol = 1e-8
    nsets = {
        'base': np.where(abs(nodos[:, 1]) < tol)[0],
        'aguas_arriba': np.where(abs(nodos[:, 0] - [presa.x_up(y) for y in nodos[:, 1]]) < tol)[0],
        'aguas_abajo': np.where(abs(nodos[:, 0] - [presa.x_down(y) for y in nodos[:, 1]]) < tol)[0],
        'coronamiento': np.where(abs(nodos[:, 1] - presa.Hd) < tol)[0],
    }
    return nodos, conn, nsets
