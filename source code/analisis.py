"""analisis.py .flujo completo de la presa (lectura -> cargas -> solucion -> resultados).

    res = analizar_presa('mallas/presa_Q4.txt', bc='empotrada')
"""
import numpy as np

from .parser import leer_malla
from .ensamble import ensamblar_K
from .fisica import (fuerzas_volumen, fuerzas_presion, presion_hidrostatica, restricciones_base,
                     aristas_en_nset, resortes_robin)
from .solver import resolver, reacciones_robin
from .postproceso import (tensiones_nodales, principales, momentos_vuelco)


def analizar_presa(ruta, bc='empotrada', kn=None, kt=None, modo_robin='diagonal', borde_libre=2.0,
                   H=None, malla=None):
    """Resuelve la presa.

    bc      : 'empotrada' (u=v=0), 'deslizante' (v=0) o 'robin' (resortes kn, kt en la base).
    H       : altura del agua; por defecto (altura de la malla - borde_libre).
    malla   : (mesh, properties, nsets) ya leida (para no releer el archivo).
    """
    mesh, properties, nsets = malla if malla is not None else leer_malla(ruta)
    nodos = mesh['nodes']['position']
    Hd = nodos[:, 1].max()
    H = Hd - borde_libre if H is None else H
    base = nsets['base']

    K = ensamblar_K(mesh, properties)
    F_peso = fuerzas_volumen(mesh, properties)
    ar_up = aristas_en_nset(mesh, nsets['aguas_arriba'])
    F_agua = fuerzas_presion(mesh, properties, ar_up, presion_hidrostatica(H))
    F = F_peso + F_agua

    Kextra = None
    if bc == 'robin':
        ar_base = aristas_en_nset(mesh, base)
        Kextra = resortes_robin(mesh, properties, ar_base, kn, kn if kt is None else kt, modo_robin)
        fixed = np.zeros(F.size, bool)                     # no hay GDL restringidos
        D, _ = resolver(K, F, fixed, Kextra=Kextra)
        R = reacciones_robin(Kextra, D)                    # fuerza del suelo sobre la presa
    else:
        fixed = restricciones_base(mesh, base, bc)
        D, R = resolver(K, F, fixed)

    # Dn: desplazamientos por nodo (nNodos x nDofNod), como res.Dn del MATLAB
    Dn = D[mesh['nodes']['dof']]
    sig = tensiones_nodales(mesh, properties, D)
    s1, s2, ang = principales(sig)
    mom = momentos_vuelco(mesh, properties, base, R, H, nodos_aguas_arriba=nsets['aguas_arriba'])
    return {'mesh': mesh, 'properties': properties, 'nsets': nsets, 'K': K, 'F': F,
            'F_peso': F_peso, 'F_agua': F_agua, 'D': D, 'Dn': Dn, 'R': R, 'sigma': sig, 'sigma1': s1,
            'sigma2': s2, 'H': H, 'Hd': Hd, 'momentos': mom, 'bc': bc, 'Kextra': Kextra}


def resumen(res):
    """Texto con los resultados principales."""
    m, D, nodos = res['momentos'], res['D'], res['mesh']['nodes']['position']
    dof = res['mesh']['nodes']['dof']
    cor = res['nsets']['coronamiento']
    ux_c = D[dof[cor, 0]].max()
    s1 = res['sigma1']
    g = res['F_peso'].reshape(-1, 2)[:, 1].sum()
    a = res['F_agua'].reshape(-1, 2).sum(axis=0)
    L = [f"Condicion de base       : {res['bc']}",
         f"Nodos / elementos       : {nodos.shape[0]} / {len(res['mesh']['elements']['connectivity'])}",
         f"Area A                  : {m['A']:.3f} m2     peso W = {m['peso'] / 1e6:.4f} MN/m",
         f"Peso propio (nodal)     : {-g / 1e6:.4f} MN/m",
         f"Empuje agua (nodal)     : Fx = {a[0] / 1e6:.4f} MN/m (analitico {m['empuje'] / 1e6:.4f})",
         f"Desplazamiento max x cor: {ux_c * 1e3:.4f} mm",
         f"sigma_1 max / min       : {s1.max() / 1e6:.3f} / {s1.min() / 1e6:.3f} MPa",
         f"M_estabilizante (anal.) : {m['M_est'] / 1e6:.3f} MN m/m",
         f"M_desestabiliz. (anal.) : {m['M_des'] / 1e6:.3f} MN m/m",
         f"FS vuelco = M_est/M_des : {m['FS']:.3f}" + (f"   (sin peso del agua: {m['FS_sin_agua']:.3f})" if m['mu'] > 0 else ''),
         f"M_reaccion FEM (pie)    : {m['M_R'] / 1e6:.3f} MN m/m   (esperado {-m['M_neto'] / 1e6:.3f})",
         f"Reaccion  Rx / Ry       : {m['Rx_tot'] / 1e6:.4f} / {m['Ry_tot'] / 1e6:.4f} MN/m",
         f"Reaccion vertical en x  : {m['x_res']:.3f} m  (tercio medio: "
         f"{m['x_talon'] + m['B'] / 3:.2f} a {m['x_talon'] + 2 * m['B'] / 3:.2f} m) -> "
         f"{'dentro' if m['tercio_medio'] else 'FUERA'}"]
    return '\n'.join(L)
