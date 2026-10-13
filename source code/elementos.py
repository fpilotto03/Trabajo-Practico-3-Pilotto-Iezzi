"""elementos.py  --   (motor FEM): librería de elementos planos.

Contiene la matemática LOCAL de cada elemento, tal como se vio en clase (clases 7, 8 y 9):

    - funciones de forma N(xi, eta) y sus derivadas respecto de (xi, eta)
    - jacobiano J = dN/d(xi,eta) * X_e   y   det(J)
    - matriz B (3 x 2n) con las derivadas respecto de (x, y):  [dN/dx; dN/dy] = J^-1 [dN/dxi; dN/deta]
    - matriz constitutiva D (estado plano de tension)
    - reglas de integracion de Gauss
    - matriz de rigidez del elemento   Ke = sum_g  w_g * B' * D * B * det(J) * t

Elementos soportados (numeracion de nodos de Cook, igual a la que exporta el mallador NX):

    T3 (CST, 3 nodos)   T6 (LST, 6 nodos)   Q4 (4 nodos)   Q8 (8 nodos)   Q9 (9 nodos)

            4---7---3                          3
            |       |                          | \\
            8   9   6      (Q4, Q8, Q9)        6   5      (T3, T6)
            |       |                          |     \\
            1---5---2                          1---4---2

Los vertices se numeran en sentido ANTIHORARIO y despues vienen los nodos medios (1-2, 2-3, 3-4, 4-1)
y, solo en Q9, el nodo central.  Por eso siempre se verifica  det(J) > 0.

Vector de desplazamientos del elemento (como en MATLAB):  de = [u1 v1 u2 v2 ... un vn]'
Vector de deformaciones:  eps = [eps_x, eps_y, gamma_xy]'   (gamma_xy = 2*eps_xy)

Contrato de interfaz (ver docs/arquitectura_interfaces.md):
    obtener_puntos_gauss(tipo)                   -> puntos (n,2), pesos (n,)
    obtener_N_y_detJ(xy_e, tipo, xi, eta)        -> N (n,), detJ
    obtener_B_y_D(xy_e, prop, tipo, xi, eta)     -> B (3,2n), D (3,3)
"""
import numpy as np

# --------------------------------------------------------------------------------------------
# Tabla de elementos
# --------------------------------------------------------------------------------------------
# Coordenadas naturales (xi, eta) de los nodos de cada elemento.
NODOS_NAT = {
    'T3': np.array([[0, 0], [1, 0], [0, 1]], float),
    'T6': np.array([[0, 0], [1, 0], [0, 1], [.5, 0], [.5, .5], [0, .5]], float),
    'Q4': np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float),
    'Q8': np.array([[-1, -1], [1, -1], [1, 1], [-1, 1],
                    [0, -1], [1, 0], [0, 1], [-1, 0]], float),
    'Q9': np.array([[-1, -1], [1, -1], [1, 1], [-1, 1],
                    [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]], float),
}
N_NODOS = {t: len(v) for t, v in NODOS_NAT.items()}
ORDEN = {'T3': 1, 'T6': 2, 'Q4': 1, 'Q8': 2, 'Q9': 2}      # orden de interpolacion
ES_TRIANGULO = {'T3': True, 'T6': True, 'Q4': False, 'Q8': False, 'Q9': False}

# Aristas de cada elemento (indices locales 0-based): (nodo_a, nodo_b [, nodo_medio]),
# recorridas en sentido antihorario (la normal exterior es (ty, -tx)).  Las usa el ROL 2
# para las cargas de superficie (presion hidrostatica).
ARISTAS = {
    'T3': [(0, 1), (1, 2), (2, 0)],
    'T6': [(0, 1, 3), (1, 2, 4), (2, 0, 5)],
    'Q4': [(0, 1), (1, 2), (2, 3), (3, 0)],
    'Q8': [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
    'Q9': [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
}


def _verificar_tipo(tipo):
    if tipo not in NODOS_NAT:
        raise ValueError(f"Tipo de elemento desconocido: '{tipo}'. Validos: {list(NODOS_NAT)}")


# --------------------------------------------------------------------------------------------
# Funciones de forma
# --------------------------------------------------------------------------------------------
def _lagrange_1d(t):
    """Polinomios de Lagrange cuadraticos en los nodos t = -1, 0, 1 y sus derivadas (para Q9)."""
    L = np.array([t * (t - 1) / 2, 1 - t * t, t * (t + 1) / 2])
    dL = np.array([(2 * t - 1) / 2, -2 * t, (2 * t + 1) / 2])
    return L, dL


def funciones_forma(tipo, xi, eta):
    """Funciones de forma N (n,) y derivadas dN (2,n) = [dN/dxi ; dN/deta] en el punto (xi, eta).

    Condiciones vistas en clase:  Ni(nodo j) = delta_ij  y  sum(Ni) = 1.
    """
    _verificar_tipo(tipo)
    if tipo == 'T3':                                    # CST: u = a1 + a2 x + a3 y
        N = np.array([1 - xi - eta, xi, eta])
        dN = np.array([[-1.0, 1.0, 0.0],
                       [-1.0, 0.0, 1.0]])

    elif tipo == 'T6':                                  # LST: polinomio completo de grado 2
        L = 1 - xi - eta
        N = np.array([L * (2 * L - 1), xi * (2 * xi - 1), eta * (2 * eta - 1),
                      4 * L * xi, 4 * xi * eta, 4 * eta * L])
        dN = np.array([[-(4 * L - 1), 4 * xi - 1, 0.0, 4 * (L - xi), 4 * eta, -4 * eta],
                       [-(4 * L - 1), 0.0, 4 * eta - 1, -4 * xi, 4 * xi, 4 * (L - eta)]])

    elif tipo == 'Q4':                                  # N_i = 1/4 (1 + xi_i xi)(1 + eta_i eta)
        xn, en = NODOS_NAT['Q4'][:, 0], NODOS_NAT['Q4'][:, 1]
        N = 0.25 * (1 + xn * xi) * (1 + en * eta)
        dN = np.array([0.25 * xn * (1 + en * eta),
                       0.25 * en * (1 + xn * xi)])

    elif tipo == 'Q8':                                  # serendipito
        N = np.zeros(8)
        dN = np.zeros((2, 8))
        for i, (xn, en) in enumerate(NODOS_NAT['Q8']):
            a, b = xn * xi, en * eta
            if i < 4:                                   # vertices
                N[i] = 0.25 * (1 + a) * (1 + b) * (a + b - 1)
                dN[0, i] = 0.25 * xn * (1 + b) * (2 * a + b)
                dN[1, i] = 0.25 * en * (1 + a) * (a + 2 * b)
            elif xn == 0:                               # medio de arista (eta = +-1)
                N[i] = 0.5 * (1 - xi ** 2) * (1 + b)
                dN[0, i] = -xi * (1 + b)
                dN[1, i] = 0.5 * (1 - xi ** 2) * en
            else:                                       # medio de arista (xi = +-1)
                N[i] = 0.5 * (1 + a) * (1 - eta ** 2)
                dN[0, i] = 0.5 * xn * (1 - eta ** 2)
                dN[1, i] = -eta * (1 + a)

    else:                                               # Q9: producto de Lagrange 1D
        Lx, dLx = _lagrange_1d(xi)
        Le, dLe = _lagrange_1d(eta)
        idx = {-1.0: 0, 0.0: 1, 1.0: 2}
        N = np.zeros(9)
        dN = np.zeros((2, 9))
        for i, (xn, en) in enumerate(NODOS_NAT['Q9']):
            ix, ie = idx[float(xn)], idx[float(en)]
            N[i] = Lx[ix] * Le[ie]
            dN[0, i] = dLx[ix] * Le[ie]
            dN[1, i] = Lx[ix] * dLe[ie]
    return N, dN


# --------------------------------------------------------------------------------------------
# Jacobiano y matriz B (formulacion isoparametrica, clase 8)
# --------------------------------------------------------------------------------------------
def jacobiano(xy_e, dN):
    """J = [ sum N,xi * x   sum N,xi * y ; sum N,eta * x   sum N,eta * y ]  (2x2) y det(J).

    xy_e : coordenadas nodales del elemento (n,2).   dN : (2,n) derivadas en coordenadas naturales.
    """
    J = dN @ xy_e
    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise ValueError('det(J) <= 0: revisar el orden de los nodos (antihorario, convencion de '
                         'Cook) o la distorsion del elemento')
    return J, detJ


def matriz_B(xy_e, tipo, xi, eta):
    """Matriz B (3 x 2n) en el punto (xi, eta) y det(J).

        [dN/dx]       [dN/dxi ]                      | N1,x  0     N2,x  0    ... |
        [dN/dy] = J^-1[dN/deta]      B(x,y)    =     | 0     N1,y  0     N2,y ... |
                                                     | N1,y  N1,x  N2,y  N2,x ... |
    """
    N, dN = funciones_forma(tipo, xi, eta)
    J, detJ = jacobiano(xy_e, dN)
    dNxy = np.linalg.solve(J, dN)                      # J^-1 * dN   -> (2,n)
    n = dNxy.shape[1]
    B = np.zeros((3, 2 * n))
    B[0, 0::2] = dNxy[0]                               # eps_x  = u,x
    B[1, 1::2] = dNxy[1]                               # eps_y  = v,y
    B[2, 0::2] = dNxy[1]                               # gamma  = u,y + v,x
    B[2, 1::2] = dNxy[0]
    return B, detJ


# --------------------------------------------------------------------------------------------
# Matriz constitutiva (clase 7)
# --------------------------------------------------------------------------------------------
def matriz_D(E, nu, estado='tension_plana'):
    """Matriz constitutiva D (3x3) con eps = [eps_x eps_y gamma_xy]'.

    tension_plana    (sigma_z = 0):  E/(1-nu^2) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2]
    deformacion_plana (eps_z = 0) :  E/((1+nu)(1-2nu)) * [1-nu nu 0; nu 1-nu 0; 0 0 (1-2nu)/2]
    """
    if estado == 'tension_plana':
        return E / (1 - nu ** 2) * np.array([[1, nu, 0],
                                             [nu, 1, 0],
                                             [0, 0, (1 - nu) / 2]])
    if estado == 'deformacion_plana':
        c = E / ((1 + nu) * (1 - 2 * nu))
        return c * np.array([[1 - nu, nu, 0],
                             [nu, 1 - nu, 0],
                             [0, 0, (1 - 2 * nu) / 2]])
    raise ValueError("estado debe ser 'tension_plana' o 'deformacion_plana'")


def D_de_prop(prop):
    """Matriz D a partir de un diccionario de propiedades {'E', 'nu', ['estado']}."""
    return matriz_D(prop['E'], prop['nu'], prop.get('estado', 'tension_plana'))


# --------------------------------------------------------------------------------------------
# Cuadratura de Gauss (clase 8: integracion numerica)
# --------------------------------------------------------------------------------------------
def _gauss_cuad(n):
    """Regla producto n x n sobre [-1,1]^2 -> puntos (n^2,2), pesos (n^2,)."""
    x, w = np.polynomial.legendre.leggauss(n)
    pts = np.array([[x[i], x[j]] for j in range(n) for i in range(n)])
    pesos = np.array([w[i] * w[j] for j in range(n) for i in range(n)])
    return pts, pesos


def obtener_puntos_gauss(tipo, para='rigidez'):
    """Puntos de integracion naturales (n,2) y pesos (n,) del elemento.

    para='rigidez'  : regla usada para integrar Ke y las fuerzas de volumen
                      Q4: 2x2   Q8/Q9: 3x3   T3: 1 punto   T6: 3 puntos  (los pesos de los
                      triangulos suman 1/2 = area del triangulo de referencia).
    para='tensiones': puntos de muestreo de tensiones para extrapolar a los nodos (clase 9):
                      cuadrilateros 2x2 (superconvergentes), triangulos igual que rigidez.
    """
    _verificar_tipo(tipo)
    if tipo == 'T3':
        return np.array([[1 / 3, 1 / 3]]), np.array([0.5])
    if tipo == 'T6':
        return np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]), np.full(3, 1 / 6)
    if tipo == 'Q4' or para == 'tensiones':
        return _gauss_cuad(2)
    return _gauss_cuad(3)                                # Q8, Q9


# --------------------------------------------------------------------------------------------
# Matriz de rigidez del elemento
# --------------------------------------------------------------------------------------------
def matriz_Kele(xy_e, tipo, prop):
    """Ke = sum_g w_g * B' D B * det(J) * t   (2n x 2n).  prop = {'E','nu','t',['estado']}."""
    D = D_de_prop(prop)
    t = prop.get('t', 1.0)
    pts, pesos = obtener_puntos_gauss(tipo, 'rigidez')
    n = xy_e.shape[0]
    Ke = np.zeros((2 * n, 2 * n))
    for (xi, eta), w in zip(pts, pesos):
        B, detJ = matriz_B(xy_e, tipo, xi, eta)
        Ke += w * (B.T @ D @ B) * detJ * t
    return Ke


# --------------------------------------------------------------------------------------------
# Funciones de contrato con el ROL 2
# --------------------------------------------------------------------------------------------
def obtener_N_y_detJ(xy_e, tipo, xi, eta):
    """Vector N (n,) y escalar det(J) en (xi, eta).  Sirve para integrar el peso propio:
    r_e = sum_g w_g * N' * (rho g) * det(J) * t."""
    N, dN = funciones_forma(tipo, xi, eta)
    _, detJ = jacobiano(np.asarray(xy_e, float), dN)
    return N, detJ


def obtener_B_y_D(xy_e, prop, tipo, xi, eta):
    """Matriz B (3,2n) y matriz constitutiva D (3,3) en (xi, eta).  Tension: sigma = D B de."""
    B, _ = matriz_B(np.asarray(xy_e, float), tipo, xi, eta)
    return B, D_de_prop(prop)
