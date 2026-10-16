# Contratos de interfaz 

El motor FEM entrega `src/elementos.py`, `src/ensamble.py` y `patch_test.py`.
Todo se importa desde la raíz del repo (`from src.elementos import ...`). 

- **Nodos de cada elemento**: convención de Cook (la que exporta NX), antihorarios: vértices primero, luego
  nodos medios (1-2, 2-3, 3-4, 4-1) y, solo en Q9, el centro. Se valida `det(J) > 0`.
- **Tipos**: `'T3'`, `'T6'`, `'Q4'`, `'Q8'`, `'Q9'`.
- **Índices 0-based**. Grados de libertad: nodo `i` → `u = 2*i`, `v = 2*i + 1`
  (igual que `mesh['nodes']['dof'][i, :]`).
- **Propiedades**: diccionario `{'E': 21e9, 'nu': 0.2, 't': 1.0, 'rho': 2400.0}`
  (opcional `'estado': 'tension_plana'` (por defecto) o `'deformacion_plana'`).
- **Vectores**: `de = [u1 v1 u2 v2 ... un vn]`, `eps = [eps_x, eps_y, gamma_xy]`, `sigma = [sx, sy, txy]`.

## Estructura `mesh` 

```python
mesh['nodes']['position']          # (nNodos, 2)
mesh['nodes']['dof']               # (nNodos, 2)  numero de GDL de u y v
mesh['elements']['connectivity']   # lista de arrays con los nodos de cada elemento
mesh['elements']['type']           # lista de strings
mesh['elements']['property']       # (nEle,) indice dentro de la lista `properties`
```

Se crea con `crear_mesh(nodos, conectividad, tipo_elemento, propiedad_elem=None)`.



| # | Función | Entrada | Salida |
|---|---|---|---|
| 1 | `armar_K_global(nodos, conectividad, propiedades, tipo_elemento)` (`ensamble.py`) | arrays del parser, propiedades y tipo | `K` global `scipy.sparse.csr_matrix` de `2·nNodos × 2·nNodos` |
| 2 | `obtener_N_y_detJ(xy_e, tipo, xi, eta)` (`elementos.py`) | coordenadas del elemento `(n,2)` y punto natural | `N` `(n,)` y escalar `detJ` |
| 3 | `obtener_B_y_D(xy_e, prop, tipo, xi, eta)` (`elementos.py`) | coordenadas, propiedades, tipo y punto natural | `B` `(3, 2n)` y `D` `(3, 3)` |
| 4 | `obtener_puntos_gauss(tipo, para='rigidez')` (`elementos.py`) | tipo (y `'tensiones'` para el muestreo de tensiones) | `puntos` `(n,2)` y `pesos` `(n,)` |

Reglas de Gauss (`para='rigidez'`): Q4 2×2, Q8/Q9 3×3, T3 1 punto, T6 3 puntos.
Con `para='tensiones'` los cuadriláteros usan 2×2 (puntos de muestreo para extrapolar a los nodos, clase 9).
Los pesos de los triángulos suman 1/2 y los de los cuadriláteros 4.

Extras que expone `elementos.py`: `funciones_forma(tipo, xi, eta)` → `N`, `dN` (2×n); `matriz_B(...)`;
`matriz_D(E, nu, estado)`; `matriz_Kele(xy_e, tipo, prop)`; `NODOS_NAT[tipo]` (coordenadas naturales de los nodos,
útiles para evaluar tensiones en los nodos); `ARISTAS[tipo]` (aristas locales `(a, b[, medio])`, antihorarias,
para las cargas de superficie).



```python
import numpy as np
from src.elementos import obtener_N_y_detJ, obtener_B_y_D, obtener_puntos_gauss
from src.ensamble import armar_K_global

K = armar_K_global(nodos, conectividad, prop, tipo)                 # 1

# Peso propio:  r_e = sum_g w_g N' (rho g) detJ t
F = np.zeros(2 * len(nodos))
puntos, pesos = obtener_puntos_gauss(tipo)                          # 4
for c in conectividad:
    xy_e = nodos[c, :]
    for (xi, eta), w in zip(puntos, pesos):
        N, detJ = obtener_N_y_detJ(xy_e, tipo, xi, eta)             # 2
        F[2 * c + 1] += -w * N * prop['rho'] * 9.81 * detJ * prop['t']

# Tensiones en los puntos de Gauss:  sigma = D B de
B, D = obtener_B_y_D(xy_e, prop, tipo, xi, eta)                     # 3
sigma = D @ (B @ de)
```

## Patch test

`python patch_test.py` verifica los 5 elementos con 4 estados de tensión constante y dos variantes
(cargas en el contorno con apoyos isostáticos, y desplazamientos impuestos en el contorno). Error esperado ≈ 1e-14.
`python -m unittest discover -s tests -t .` corre además los tests de las funciones de forma, rigidez
(modos de cuerpo rígido, `T3` contra la fórmula de clase) y ensamble.
