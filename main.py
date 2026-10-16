
"""main.py  --  Llama a las funciones de los modulos.

Uso:
    python main.py                                   # presa Q8, base empotrada
    python main.py --malla mallas/presa_T3.txt --escala 3000
    python main.py --bc robin --k 5e10 --modo diagonal
    python main.py --todas                           # las 5 mallas + tabla comparativa
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.analisis import analizar_presa, resumen
from src.postproceso import graficar_deformada, graficar_sigma1, tension_en_base


def figuras(res, nombre, escala, salida):
    os.makedirs(salida, exist_ok=True)
    mesh, props = res['mesh'], res['properties']
    graficar_deformada(mesh, res['D'], escala, f'{salida}/{nombre}_deformada.png')
    graficar_sigma1(mesh, props, res['D'], escala, f'{salida}/{nombre}_sigma1.png')
    x, sy, txy, s1 = tension_en_base(mesh, props, res['D'], res['nsets']['base'])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, sy / 1e6, 'o-', label=r'$\sigma_y$ base')
    ax.plot(x, s1 / 1e6, 's--', label=r'$\sigma_1$ base')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('x [m]  (talon = 0)')
    ax.set_ylabel('tension [MPa]')
    ax.set_title(f'Tensiones en la base - {nombre}')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(f'{salida}/{nombre}_base.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--malla', default='mallas/presa_Q8.txt')
    ap.add_argument('--bc', default='empotrada', choices=['empotrada', 'deslizante', 'robin'])
    ap.add_argument('--k', type=float, default=5e10, help='rigidez de la fundacion kn [N/m3] (robin)')
    ap.add_argument('--kt', type=float, default=None, help='rigidez tangencial (por defecto = kn)')
    ap.add_argument('--modo', default='diagonal', choices=['diagonal', 'consistente'])
    ap.add_argument('--escala', type=float, default=2000.0, help='factor de escala de la deformada')
    ap.add_argument('--salida', default='results')
    ap.add_argument('--todas', action='store_true')
    a = ap.parse_args()

    mallas = ([f'mallas/presa_{t}.txt' for t in ('T3', 'T6', 'Q4', 'Q8', 'Q9')] if a.todas else [a.malla])
    filas = []
    for ruta in mallas:
        nombre = os.path.splitext(os.path.basename(ruta))[0] + ('' if a.bc == 'empotrada' else '_' + a.bc)
        res = analizar_presa(ruta, bc=a.bc, kn=a.k, kt=a.kt, modo_robin=a.modo)
        print(f'=== {ruta} ===')
        print(resumen(res), '\n')
        figuras(res, nombre, a.escala, a.salida)
        m = res['momentos']
        filas.append((nombre, res['D'][res['mesh']['nodes']['dof'][res['nsets']['coronamiento'], 0]].max() * 1e3,
                      res['sigma1'].max() / 1e6, m['FS'], m['M_R'] / 1e6))
    if a.todas:
        print(f"{'malla':<12}{'u_x cor [mm]':>14}{'s1max [MPa]':>14}{'FS':>8}{'M_R [MNm/m]':>14}")
        for f in filas:
            print(f'{f[0]:<12}{f[1]:>14.4f}{f[2]:>14.3f}{f[3]:>8.3f}{f[4]:>14.3f}')
    print('Figuras en', a.salida)


if __name__ == '__main__':
    main()
