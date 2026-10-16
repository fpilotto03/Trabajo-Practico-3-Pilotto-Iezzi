# Fisica, post-proceso y presa de gravedad

Resultados obtenidos con `python main.py --todas`, `estudio_diseno.py` y `estudio_robin.py`
(por metro de profundidad, **deformacion plana** (eps_z = 0, segun correccion del titulo del TP), E = 21 GPa, nu = 0.2, rho_c = 2400, rho_w = 1000, g = 9.81).

> **Nota:** el enunciado fue corregido de "estado plano de tension" a "estado plano de deformacion". El modelo usa D de deformacion plana (`plane_strain` en las mallas); FS y equilibrio (cargas y reacciones) no dependen del estado plano, solo cambian ligeramente desplazamientos y tensiones. Con la presa larga en la direccion perpendicular, deformacion plana es la hipotesis fisicamente adecuada.

## 1. Cargas (`src/fisica.py`)
* **Peso propio**: `F_e = sum_g w_g N^T (rho g) detJ t` con las funciones de contrato (`obtener_puntos_gauss`, `obtener_N_y_detJ`). Total = 7.4164 MN/m (A = 315 m2).
* **Presion hidrostatica** `p = rho_w g (H - y)`, H = 28 m (borde libre 2 m), sobre las aristas del paramento aguas arriba: traccion `-p n`, integrada con 3 puntos de Gauss 1D y funciones de forma lineales/cuadraticas de la arista. Resultante Fx = 3.8455 MN/m = rho_w g H^2/2 (exacta), con la fila de nodos forzada a y = H.
* **Base empotrada (u = v = 0)**: hipotesis principal. Corresponde a una fundacion de roca sana infinitamente rigida respecto del hormigon (E_roca >> E_hormigon = 21 GPa) y perfectamente adherida: ni desliza ni se despega ni rota. Es valida mientras (i) la rigidez de la fundacion sea mucho mayor que la de la presa, (ii) el corte en la base no supere la resistencia por friccion/cohesion (tau <= c + sigma tan(phi)) y (iii) no haya traccion que despegue la junta. Con suelo impermeable no hay subpresion, por eso no se la incluye. Su costo: impone una singularidad de tensiones en el talon y el pie (esquinas), donde sigma_1 depende de la malla.
* **Base deslizante (v = 0 + un nodo fijo en x)**: corresponde a apoyo sin friccion/cohesion; descarga todo el empuje en un nodo y da tensiones espurias (sigma_1 ~ 9.7 MPa, calculado en deformacion plana), por eso solo se usa como contraste, no para el diseno.
* **Robin (bonus)** = fundacion elastica tipo Winkler (resortes distribuidos): corresponde a un suelo deformable de rigidez finita; k -> infinito recupera la empotrada y k -> 0 la base libre.   `t = -(kn u_n n + kt u_t tau)`. `modo='diagonal'` suma `w_i (kn n n^T + kt tau tau^T)` a la diagonal con la longitud tributaria (1/2,1/2 lineal; 1/6,1/6,4/6 cuadratico); `modo='consistente'` integra `N^T k N`. Diferencias < 0.1 % en desplazamientos; para k -> infinito se recupera la base empotrada.

## 2. Resultados (base empotrada)
| Malla | u_x corona [mm] | sigma_1 max [MPa] | FS | M_R FEM [MN m/m] |
|---|---|---|---|---|
| T3 (480 el.) | 0.650 | 0.267 | 2.450 | -52.045 |
| T6 (300 el.) | 0.664 | 0.475 | 2.450 | -52.045 |
| Q4 (240 el.) | 0.661 | 0.621 | 2.450 | -52.045 |
| Q8 (150 el.) | 0.664 | 0.631 | 2.450 | -52.045 |
| Q9 (150 el.) | 0.664 | 0.623 | 2.450 | -52.045 |

Los desplazamientos convergen (~0.66 mm); **sigma_1 maximo no converge**: ocurre en el talon (x = 0, y = 0) donde hay singularidad (esquina con base empotrada), y crece al refinar. Las tensiones fuera de la esquina (sigma_y en la base entre -0.38 y -0.55 MPa, compresion) son estables.

## 3. Momentos respecto del pie (`postproceso.momentos_vuelco`)
* Analitico: `M_est = rho_c g A (x_pie - x_G) = 87.94 MN m/m` (x_G = 6.143 m, brazo 11.857 m); `M_des = rho_w g H^3/6 = 35.89 MN m/m`; **FS = 2.45**.
* FEM: `M_R = sum[(x - x_pie) Ry - y Rx]` sobre los nodos de la base = **-52.045 MN m/m = -(M_est - M_des)** (error ~1e-12): la reaccion equilibra el par neto. Rx = -3.8455 y Ry = 7.4164 MN/m.
* La reaccion vertical pasa por x = 10.98 m desde el talon, dentro del tercio medio (6 a 12 m): no hay traccion global en la base. FS > 1.5 y resultante en el tercio medio -> vuelco verificado (la tension local de traccion en el talon es un efecto de la singularidad).

## 4. Variaciones de geometria (`estudio_diseno.py`, malla Q8)
| Caso | FS | x reaccion [m] | tercio medio |
|---|---|---|---|
| base (B=18, c=3) | 2.45 | 10.98 | si |
| B = 20 / 22 / 24 | 2.99 / 3.58 / 4.22 | 11.22 / 11.52 / 11.88 | si |
| c = 4 / 5 | 2.55 / 2.63 | 10.86 / 10.78 | si |
| mu = 0.1 / 0.2 (aguas arriba inclinado, B=18) | 2.40 / 2.32 | 11.57 / 12.20 | si / no |
| B = 22, mu = 0.1 | 3.53 | 12.15 | si |

Ensanchar la base (menor pendiente aguas abajo) es lo mas efectivo: el momento estabilizante crece ~B^2 mientras que M_des no cambia. Ampliar la corona ayuda poco. Inclinar el paramento aguas arriba a B fijo *reduce* FS (se quita peso de hormigon; el peso del agua sobre el paramento, que se incluye, no lo compensa).

## 5. Fundacion elastica (`estudio_robin.py`)
Con k = 2e10 N/m3 (diagonal) sigma_1 max pasa de 0.631 a 0.353 MPa y sigma_y en el talon de +0.41 a +0.23 MPa; con k = 1e9 baja a 0.14 MPa pero el desplazamiento de la corona sube de 0.66 a 1.86 mm. FS y M_R no cambian (son equilibrio global). La fundacion blanda reparte la reaccion y atenua la singularidad.
