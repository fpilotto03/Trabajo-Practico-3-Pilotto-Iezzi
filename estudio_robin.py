"""Bonus: fundacion elastica (Robin) en la base.  Barre kn y compara con la base empotrada."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analisis import analizar_presa
from src.postproceso import tension_en_base

if __name__ == '__main__':
    ruta = 'mallas/presa_Q8.txt'
    ref = analizar_presa(ruta)
    ks = [1e9, 5e9, 2e10, 1e11, 1e12, 1e14]
    print(f"{'k [N/m3]':>10}{'modo':>12}{'ux cor [mm]':>13}{'s1max':>8}{'FS':>7}{'M_R [MNm]':>11}")
    fig, ax = plt.subplots(figsize=(7, 4))
    filas = []
    for k in ks:
        for modo in ('diagonal', 'consistente'):
            r = analizar_presa(ruta, bc='robin', kn=k, modo_robin=modo)
            ux = r['D'][r['mesh']['nodes']['dof'][r['nsets']['coronamiento'], 0]].max() * 1e3
            filas.append([k, modo, ux, r['sigma1'].max() / 1e6, r['momentos']['FS'], r['momentos']['M_R'] / 1e6])
            print(f"{k:>10.0e}{modo:>12}{ux:>13.4f}{r['sigma1'].max() / 1e6:>8.3f}"
                  f"{r['momentos']['FS']:>7.3f}{r['momentos']['M_R'] / 1e6:>11.3f}")
        if k in (1e9, 2e10, 1e12):
            x, sy, _, _ = tension_en_base(r['mesh'], r['properties'], r['D'], r['nsets']['base'])
            ax.plot(x, sy / 1e6, label=f'Robin k={k:.0e}')
    x, sy, _, _ = tension_en_base(ref['mesh'], ref['properties'], ref['D'], ref['nsets']['base'])
    ax.plot(x, sy / 1e6, 'k--', label='empotrada')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('sigma_y en la base [MPa]')
    ax.legend()
    ax.grid(alpha=.3)
    fig.savefig('results/estudio_robin.png', dpi=140, bbox_inches='tight')
    with open('results/estudio_robin.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['k', 'modo', 'ux_cor_mm', 'sigma1_max_MPa', 'FS', 'M_R_MNm'])
        w.writerows(filas)
