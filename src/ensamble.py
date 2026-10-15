"""ensamble.py  --  ROL 1 (motor FEM): ensamble global de la matriz de rigidez.

Misma logica que el codigo de MATLAB del TP1 (resolver_tijera, seccion "5. Ensamble"):

    1. se arma la estructura  mesh.nodes.position / mesh.nodes.dof / mesh.elements.connectivity
    2. se calcula el total de grados de libertad  nDofTot = nNodos * nDofNod
    3. se recorre cada elemento: Kele{iEle}, rowIndex{iEle}, columnIndex{iEle}
    4. K = sparse(rows, cols, vals)   ->   scipy.sparse.coo_matrix(...).tocsr()  (suma duplicados)

Diferencias por pasar de MATLAB a Python:  los indices son 0-based y las estructuras 'struct'
anidadas de MATLAB se escriben como diccionarios:

    mesh['nodes']['position']        (nNodos, 2)   coordenadas x, y
    mesh['nodes']['dof']             (nNodos, 2)   numero de grado de libertad (u, v) de cada nodo
    mesh['elements']['connectivity'] lista de arrays (nEle) con los nodos de cada elemento
    mesh['elements']['type']         lista de strings 'T3' | 'T6' | 'Q4' | 'Q8' | 'Q9'
    mesh['elements']['property']     (nEle,)  indice (0-based) dentro de la lista `properties`

    properties = [ {'E': 21e9, 'nu': 0.2, 't': 1.0, 'rho': 2400.0}, ... ]
"""
import numpy as np
import scipy.sparse as sp

from .elementos import N_NODOS, matriz_Kele


def crear_mesh(nodos, conectividad, tipo_elemento, propiedad_elem=None):
    """Arma la estructura `mesh` (equivalente al struct de MATLAB).

    nodos          : (nNodos, 2) coordenadas.
    conectividad   : (nEle, nNodosPorElemento) array, o lista de arrays si se mezclan tipos.
                     Indices de nodos 0-based.
    tipo_elemento  : un string (todos iguales) o una lista de strings (uno por elemento).
    propiedad_elem : indice de propiedad de cada elemento (por defecto, todos 0).
    """
    nodos = np.asarray(nodos, float)
    conn = [np.asarray(c, int) for c in conectividad]
    nEle = len(conn)
    tipos = [tipo_elemento] * nEle if isinstance(tipo_elemento, str) else list(tipo_elemento)
    if len(tipos) != nEle:
        raise ValueError('tipo_elemento debe ser un string o tener un valor por elemento')
    for iEle in range(nEle):
        if len(conn[iEle]) != N_NODOS[tipos[iEle]]:
            raise ValueError(f'Elemento {iEle}: {tipos[iEle]} necesita {N_NODOS[tipos[iEle]]} nodos '
                             f'y tiene {len(conn[iEle])}')
    if conn and max(c.max() for c in conn) >= len(nodos):
        raise ValueError('La conectividad referencia un nodo inexistente')
    prop = np.zeros(nEle, int) if propiedad_elem is None else np.asarray(propiedad_elem, int)

    nDofNod = 2                                         # u, v por nodo (estado plano)
    nDofTot = nodos.shape[0] * nDofNod
    mesh = {
        'nodes': {'position': nodos,
                  # cada fila es un nodo y cada columna un grado de libertad (como en MATLAB)
                  'dof': np.arange(nDofTot).reshape(-1, nDofNod)},
        'elements': {'connectivity': conn, 'type': tipos, 'property': prop},
    }
    return mesh


def ensamblar_K(mesh, properties):
    """Matriz de rigidez global K (scipy.sparse.csr_matrix de nDofTot x nDofTot)."""
    nodos = mesh['nodes']['position']
    dof = mesh['nodes']['dof']
    conn = mesh['elements']['connectivity']
    tipos = mesh['elements']['type']
    propIdx = mesh['elements']['property']

    # Como visto en clase, el total de grados de libertad del sistema es nNodos * nDofNod.
    nDofTot = dof.size
    nEle = len(conn)

    rowIndex = [None] * nEle
    columnIndex = [None] * nEle
    Kele = [None] * nEle

    # Se itera sobre cada elemento para calcular su rigidez individual.
    for iEle in range(nEle):
        prop = properties[propIdx[iEle]]
        xy_e = nodos[conn[iEle], :]                      # coordenadas de los nodos del elemento
        Kele[iEle] = matriz_Kele(xy_e, tipos[iEle], prop)

        # Grados de libertad globales del elemento: [u1 v1 u2 v2 ... un vn]
        eleDofs = dof[conn[iEle], :].reshape(-1)
        n = eleDofs.size

        # Indices de fila y columna para saber donde poner esta matriz chica en la grande
        # (equivalente a repmat(eleDofs', 1, n) y repmat(eleDofs, n, 1) de MATLAB).
        rowIndex[iEle] = np.tile(eleDofs[:, None], (1, n))
        columnIndex[iEle] = np.tile(eleDofs[None, :], (n, 1))

    if nEle == 0:
        return sp.csr_matrix((nDofTot, nDofTot))
    # sparse(rows, cols, vals): los indices repetidos se suman (ensamble).
    K = sp.coo_matrix((np.concatenate([k.ravel() for k in Kele]),
                       (np.concatenate([r.ravel() for r in rowIndex]),
                        np.concatenate([c.ravel() for c in columnIndex]))),
                      shape=(nDofTot, nDofTot))
    return K.tocsr()


def armar_K_global(nodos, conectividad, propiedades, tipo_elemento):
    """ Devuelve K global como scipy.sparse.csr_matrix.

    nodos         : (nNodos, 2) array con x, y     (lo que lee el parser)
    conectividad  : (nEle, k) array 0-based         (k = 3, 4, 6, 8 o 9 segun el tipo)
    propiedades   : dict {'E','nu','t', ['estado']}, o lista de dicts si hay varios materiales;
                    en ese caso se usa 'property' si conectividad viene dentro de un mesh (ver
                    ensamblar_K), y aca todos los elementos usan propiedades[0].
    tipo_elemento : 'T3' | 'T6' | 'Q4' | 'Q8' | 'Q9'   (o lista de tipos)

    Numeracion de grados de libertad:  nodo i -> u = 2*i, v = 2*i + 1.
    """
    props = [propiedades] if isinstance(propiedades, dict) else list(propiedades)
    mesh = crear_mesh(nodos, conectividad, tipo_elemento)
    return ensamblar_K(mesh, props)
