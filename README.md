# TP3 - Mecanica del Solido Computacional

Analisis por elementos finitos de una **presa de gravedad de hormigon** en estado plano de deformacion
(por metro de profundidad), con elementos triangulares y cuadrilateros lineales y cuadraticos
(T3, T6, Q4, Q8, Q9), implementado en Python.

Datos del problema: E = 21 GPa, nu = 0.2, rho hormigon = 2400 kg/m3, rho agua = 1000 kg/m3, g = 9.81 m/s2.
Altura H = 30 m, corona 3 m, base 18 m, borde libre 2 m (nivel de agua a 28 m).

## Estructura

```
main.py                  director del analisis (una malla o las 5 con --todas)
generar_mallas.py        genera las mallas de la presa en mallas/*.txt
patch_test.py            patch test de los 5 elementos
estudio_diseno.py        variaciones de geometria (ancho de base, corona, pendiente aguas arriba)
estudio_robin.py         base sobre fundacion elastica (condicion de Robin)
requirements.txt         dependencias (numpy, scipy, matplotlib)
src/
  elementos.py           funciones de forma, Jacobiano, matrices B y D, puntos de Gauss, rigidez Ke
  ensamble.py            estructura de la malla y ensamble de la matriz de rigidez global (sparse)
  parser.py              lectura y escritura de mallas en texto plano
  geometria.py           geometria de la presa y generador de mallas
  fisica.py              peso propio, presion hidrostatica, condiciones de base, resortes de Robin
  solver.py              particion de grados de libertad, resolucion y reacciones
  postproceso.py         tensiones en puntos de Gauss y nodos, sigma_1, momentos, graficos
  analisis.py            flujo completo de analisis de la presa
mallas/                  mallas T3, T6, Q4, Q8, Q9, ejemplo mixto y casos de estudio (mallas/estudio)
results/                 figuras y tablas generadas
tests/                   tests unitarios (motor FEM y modelo de la presa)
docs/                    formato de malla, interfaces entre modulos y memoria de calculo
```

## Instalacion y uso

```
pip install -r requirements.txt
python patch_test.py
python generar_mallas.py
python main.py --todas
python main.py --malla mallas/presa_T3.txt --escala 3000
python main.py --bc robin --k 2e10 --escala 3000
python estudio_diseno.py
python estudio_robin.py
python -m unittest discover -s tests -t .
```

Opciones de `main.py`: `--malla` (archivo de malla), `--bc` (`empotrada`, `deslizante` o `robin`),
`--k` y `--kt` (rigidez normal y tangencial de la fundacion para Robin), `--modo` (`diagonal` o `consistente`),
`--escala` (factor de escala de la deformada), `--salida` (carpeta de resultados) y `--todas`.

## Resultados principales (base empotrada)

| Malla | u_x corona [mm] | sigma_1 max [MPa] | FS vuelco |
|---|---|---|---|
| T3 | 0.650 | 0.267 | 2.450 |
| T6 | 0.664 | 0.475 | 2.450 |
| Q4 | 0.661 | 0.621 | 2.450 |
| Q8 | 0.664 | 0.631 | 2.450 |
| Q9 | 0.664 | 0.623 | 2.450 |

* Los desplazamientos convergen (~0.66 mm). El sigma_1 maximo no converge porque aparece en el talon, donde la base empotrada genera una singularidad de tensiones.
* El peso propio (7.4164 MN/m) y el empuje hidrostatico (3.8455 MN/m) coinciden con el valor analitico, y la reaccion cierra el equilibrio de fuerzas y momentos.
* Factor de seguridad al vuelco FS = M_estabilizante / M_desestabilizante = 87.94 / 35.89 = 2.45. La reaccion vertical cae dentro del tercio medio de la base.
* Ensanchar la base es lo mas efectivo para aumentar el FS (B = 24 m da FS = 4.22).

El detalle de las hipotesis, cargas, condiciones de borde y estudios de variacion esta en
`docs/memoria_de_calculo.md`; el formato de malla en `docs/formato_malla.md` y las interfaces entre
modulos en `docs/arquitectura_interfaces.md`.
