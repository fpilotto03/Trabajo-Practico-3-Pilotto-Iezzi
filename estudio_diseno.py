"""Variaciones de la geometria para aumentar el factor de seguridad al vuelco.

FS = M_est / M_des (momentos respecto del pie).  Se varian el ancho de base B, el ancho de
corona c y la pendiente aguas arriba mu, y se verifica con FEM (malla Q8) que la reaccion
cierra el equilibrio y que la reaccion vertical queda en el tercio medio.
"""
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.geometria import Presa
from generar_mallas import escribir
from src.analisis import analizar_presa

CASOS = [('base (B=18, c=3, mu=0)', Presa()),
         ('B=20', Presa(B=20)), ('B=22', Presa(B=22)), ('B=24', Presa(B=24)),
         ('c=4', Presa(c=4, B=18)), ('c=5', Presa(c=5, B=18)),
         ('mu=0.1', Presa(mu=0.1)), ('mu=0.2', Presa(mu=0.2)),
         ('B=22, mu=0.1', Presa(B=22, mu=0.1))]


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs('mallas/estudio', exist_ok=True)
    filas = []
    print(f"{'caso':<26}{'A [m2]':>8}{'xG':>7}{'FS':>7}{'M_R+Mneto':>12}{'s1max':>8}{'x_res':>8}{'tercio':>8}")
    for i, (nom, p) in enumerate(CASOS):
        ruta = f'mallas/estudio/caso_{i}.txt'
        escribir(ruta, p, 'Q8', 6, 24, nom)
        r = analizar_presa(ruta)
        m = r['momentos']
        filas.append([nom, m['A'], m['xG'], m['FS'], (m['M_R'] + m['M_neto']) / 1e6,
                      r['sigma1'].max() / 1e6, m['x_res'] - m['x_talon'], m['tercio_medio']])
        print(f"{nom:<26}{m['A']:>8.1f}{m['xG']:>7.2f}{m['FS']:>7.2f}{filas[-1][4]:>12.1e}"
              f"{filas[-1][5]:>8.3f}{filas[-1][6]:>8.2f}{str(m['tercio_medio']):>8}")
    with open('results/estudio_diseno.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['caso', 'A', 'xG', 'FS', 'residuo_equilibrio_MNm', 'sigma1_max_MPa',
                    'x_reaccion_desde_talon', 'tercio_medio'])
        w.writerows(filas)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([f[0] for f in filas], [f[3] for f in filas], color='tab:blue')
    ax.axvline(1.5, color='r', ls='--', label='FS = 1.5')
    ax.set_xlabel('FS al vuelco')
    ax.legend()
    ax.invert_yaxis()
    fig.savefig('results/estudio_diseno.png', dpi=140, bbox_inches='tight')
