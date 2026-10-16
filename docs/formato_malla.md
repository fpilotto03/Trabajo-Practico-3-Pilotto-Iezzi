# Formato de malla de texto plano (`src/parser.py`)


```
*TITLE   Presa de gravedad - Q8
*MATERIAL hormigon plane_strain E=21e9 nu=0.2 rho=2400 t=1
*NODES
# id  x  y
1  0.0  0.0
2  3.0  0.0
*ELEMENTS
# id  TIPO  orden  material  nodos (orden Cook: esquinas antihorario, luego medios 1-2,2-3,3-4,4-1; Q9: centro al final)
1  QUAD  2  hormigon  1 2 ... 8
2  TRI   1  hormigon  4 5 6
*NSET base
1 2 3 4 5 6 7
```

| Linea de `*ELEMENTS` | Elemento |
|---|---|
| `TRI 1` + 3 nodos | T3 (triangulo lineal) |
| `TRI 2` + 6 nodos | T6 (triangulo cuadratico) |
| `QUAD 1` + 4 nodos | Q4 (cuadrilatero lineal) |
| `QUAD 2` + 8 nodos | Q8 (serendipity) |
| `QUAD 2` + 9 nodos | Q9 (lagrangiano) |

* **Tipo de elemento** y **orden de interpolacion** estan en cada linea de `*ELEMENTS`; la cantidad de nodos se verifica contra ellos. Se pueden mezclar tipos en una misma malla (ver `mallas/ejemplo_mixto.txt`).
* **Material**: `plane_stress` (tension plana) o `plane_strain` (deformacion plana), con `E`, `nu` y opcionalmente `rho` (kg/m3, para el peso propio) y `t` (espesor, por defecto 1).
* Los ids del archivo pueden ser cualquier entero (se pasan a indices 0-based).
* `*NSET nombre` define conjuntos de nodos (la presa usa `base`, `aguas_arriba`, `aguas_abajo`, `coronamiento`).
* El parser verifica: nodos inexistentes, cantidad de nodos por elemento, materiales no definidos y orientacion antihoraria (det J > 0).

```python
from src.parser import leer_malla
mesh, properties, nsets = leer_malla('mallas/presa_Q8.txt')
# mesh['nodes']['position'], mesh['nodes']['dof'], mesh['elements']['connectivity'|'type'|'property']
```
