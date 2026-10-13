"""parser.py  --  lectura y escritura del formato de malla de texto plano.

Formato (ver docs/formato_malla.md).  Todo lo que sigue a '#' es un comentario:

    *TITLE      nombre libre
    *MATERIAL   nombre  plane_stress|plane_strain  E=...  nu=...  [rho=...]  [t=...]
    *NODES
    id  x  y
    *ELEMENTS
    id  TIPO  orden  material  n1 n2 ... nk          TIPO = TRI | QUAD     orden = 1 | 2
    *NSET  nombre
    id id id ...

El tipo de elemento, el orden de interpolacion y el material se especifican en cada linea de
*ELEMENTS.  La cantidad de nodos define el elemento:  TRI 1 -> T3,  TRI 2 -> T6,  QUAD 1 -> Q4,
QUAD 2 con 8 nodos -> Q8,  QUAD 2 con 9 nodos -> Q9.

El parser devuelve los arrays equivalentes a las estructuras mesh.nodes y mesh.elements que se
usaban en MATLAB (via crear_mesh ).
"""
import numpy as np

from .ensamble import crear_mesh
from .elementos import NODOS_NAT, funciones_forma, jacobiano

ESTADOS = {'plane_stress': 'tension_plana', 'plane_strain': 'deformacion_plana'}
_ESTADO_TXT = {v: k for k, v in ESTADOS.items()}
_TIPO = {('TRI', 1, 3): 'T3', ('TRI', 2, 6): 'T6',
         ('QUAD', 1, 4): 'Q4', ('QUAD', 2, 8): 'Q8', ('QUAD', 2, 9): 'Q9'}
_TIPO_INV = {v: k for k, v in _TIPO.items()}


def leer_malla(ruta):
    """Lee el archivo y devuelve (mesh, properties, nsets).

    mesh       : estructura del ROL 1 (ver ensamble.crear_mesh); ademas
                 mesh['nodes']['ids'] y mesh['elements']['ids'] guardan los ids del archivo.
    properties : lista de dicts {'nombre','estado','E','nu','rho','t'} (mesh['elements']['property']
                 indexa esta lista)
    nsets      : dict nombre -> array de indices de nodos (0-based)
    """
    titulo, materiales, nodos_raw, elem_raw, nsets_raw = '', {}, [], [], {}
    seccion, nset_actual = None, None
    with open(ruta, encoding='utf-8') as fh:
        for nl, linea in enumerate(fh, 1):
            linea = linea.split('#', 1)[0].strip()
            if not linea:
                continue
            tok = linea.split()
            if linea.startswith('*'):
                seccion = tok[0].upper()
                if seccion == '*TITLE':
                    titulo = ' '.join(tok[1:])
                elif seccion == '*MATERIAL':
                    if len(tok) < 3 or tok[2].lower() not in ESTADOS:
                        raise ValueError(f'{ruta}:{nl}: *MATERIAL nombre plane_stress|plane_strain E=.. nu=..')
                    kv = dict(t.split('=') for t in tok[3:])
                    if 'E' not in kv or 'nu' not in kv:
                        raise ValueError(f'{ruta}:{nl}: el material necesita E y nu')
                    materiales[tok[1]] = {'nombre': tok[1], 'estado': ESTADOS[tok[2].lower()],
                                          'E': float(kv['E']), 'nu': float(kv['nu']),
                                          'rho': float(kv.get('rho', 0.0)), 't': float(kv.get('t', 1.0))}
                elif seccion == '*NSET':
                    nset_actual = tok[1]
                    nsets_raw[nset_actual] = []
                elif seccion not in ('*NODES', '*ELEMENTS'):
                    raise ValueError(f'{ruta}:{nl}: palabra clave desconocida {seccion}')
            elif seccion == '*NODES':
                nodos_raw.append((int(tok[0]), float(tok[1]), float(tok[2])))
            elif seccion == '*ELEMENTS':
                elem_raw.append((int(tok[0]), tok[1].upper(), int(tok[2]), tok[3], [int(v) for v in tok[4:]]))
            elif seccion == '*NSET':
                nsets_raw[nset_actual] += [int(v) for v in tok]
            else:
                raise ValueError(f'{ruta}:{nl}: dato fuera de una seccion')

    ids_nodos = np.array([r[0] for r in nodos_raw])
    nodos = np.array([[r[1], r[2]] for r in nodos_raw])
    indice = {nid: i for i, nid in enumerate(ids_nodos)}
    nombres = list(materiales)
    conn, tipos, prop_elem, ids_ele = [], [], [], []
    for eid, clase, orden, mat, lista in elem_raw:
        clave = (clase, orden, len(lista))
        if clave not in _TIPO:
            raise ValueError(f'Elemento {eid}: combinacion no soportada {clase} orden {orden} con {len(lista)} nodos')
        if mat not in materiales:
            raise ValueError(f"Elemento {eid}: material '{mat}' no definido")
        try:
            conn.append([indice[n] for n in lista])
        except KeyError as exc:
            raise ValueError(f'Elemento {eid}: nodo inexistente {exc}') from None
        tipos.append(_TIPO[clave]); prop_elem.append(nombres.index(mat)); ids_ele.append(eid)

    mesh = crear_mesh(nodos, conn, tipos, prop_elem)
    mesh['nodes']['ids'] = ids_nodos
    mesh['elements']['ids'] = np.array(ids_ele)
    mesh['title'] = titulo
    verificar_orientacion(mesh)
    properties = [materiales[n] for n in nombres]
    nsets = {k: np.array([indice[n] for n in v], int) for k, v in nsets_raw.items()}
    return mesh, properties, nsets


def verificar_orientacion(mesh):
    """det(J) > 0 en todos los nodos de cada elemento (orden antihorario de Cook)."""
    pos = mesh['nodes']['position']
    for iEle, (c, tipo) in enumerate(zip(mesh['elements']['connectivity'], mesh['elements']['type'])):
        for xi, eta in NODOS_NAT[tipo]:
            try:
                jacobiano(pos[c, :], funciones_forma(tipo, xi, eta)[1])
            except ValueError:
                ids = mesh['elements'].get('ids', np.arange(len(mesh['elements']['connectivity'])))
                raise ValueError(f'Elemento {ids[iEle]} ({tipo}): det(J) <= 0, orden de nodos incorrecto') from None


def escribir_malla(ruta, mesh, properties, nsets=None, titulo=None):
    """Escribe mesh/properties/nsets en el formato de texto plano."""
    ids_n = mesh['nodes'].get('ids', np.arange(1, len(mesh['nodes']['position']) + 1))
    nombres = [p['nombre'] for p in properties]
    with open(ruta, 'w', encoding='utf-8') as fh:
        fh.write(f"# Malla generada por parser.escribir_malla\n*TITLE {titulo or mesh.get('title') or 'malla'}\n")
        for p in properties:
            fh.write(f"*MATERIAL {p['nombre']} {_ESTADO_TXT[p['estado']]} E={p['E']:.12g} nu={p['nu']:.12g} "
                     f"rho={p['rho']:.12g} t={p['t']:.12g}\n")
        fh.write('*NODES\n# id x y\n')
        for i, (x, y) in enumerate(mesh['nodes']['position']):
            fh.write(f'{ids_n[i]} {x:.12g} {y:.12g}\n')
        fh.write('*ELEMENTS\n# id TIPO orden material nodos...\n')
        for i, (c, tipo) in enumerate(zip(mesh['elements']['connectivity'], mesh['elements']['type'])):
            clase, orden, _ = _TIPO_INV[tipo]
            eid = mesh['elements']['ids'][i] if 'ids' in mesh['elements'] else i + 1
            mat = nombres[mesh['elements']['property'][i]]
            fh.write(f"{eid} {clase} {orden} {mat} {' '.join(str(ids_n[n]) for n in c)}\n")
        for nombre, idx in (nsets or {}).items():
            fh.write(f'*NSET {nombre}\n')
            ids = [str(ids_n[i]) for i in idx]
            for k in range(0, len(ids), 12):
                fh.write(' '.join(ids[k:k + 12]) + '\n')
