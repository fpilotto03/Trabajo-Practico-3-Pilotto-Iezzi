"""Tests del ROL 2 (unittest):  python -m unittest discover -s tests -t ."""
import os
import tempfile
import unittest

import numpy as np

from src.parser import leer_malla, escribir_malla
from src.ensamble import crear_mesh, ensamblar_K
from src.geometria import Presa, generar_malla_presa
from src.fisica import (fuerzas_volumen, aristas_en_nset, fuerzas_presion, presion_hidrostatica,
                        restricciones_base, resortes_robin, G, RHO_AGUA)
from src.solver import resolver
from src.postproceso import (tensiones_nodales, principales, momentos_vuelco, area_y_centroide)
from src.analisis import analizar_presa
from generar_mallas import PROP, escribir

TIPOS = ['T3', 'T6', 'Q4', 'Q8', 'Q9']


def malla_presa(tipo, nx=3, ny=8, presa=None):
    nodos, conn, nsets = generar_malla_presa(presa or Presa(), tipo, nx, ny)
    return crear_mesh(nodos, conn, tipo), nsets


class TestParser(unittest.TestCase):
    def test_ida_y_vuelta_todos_los_tipos(self):
        for tipo in TIPOS:
            mesh, nsets = malla_presa(tipo)
            with tempfile.TemporaryDirectory() as d:
                ruta = os.path.join(d, 'm.txt')
                escribir_malla(ruta, mesh, [PROP], nsets, 'x')
                m2, props, ns2 = leer_malla(ruta)
            self.assertEqual(m2['elements']['type'], mesh['elements']['type'])
            np.testing.assert_allclose(m2['nodes']['position'], mesh['nodes']['position'], atol=1e-9)
            for a, b in zip(m2['elements']['connectivity'], mesh['elements']['connectivity']):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(ns2['base'], nsets['base'])
            self.assertEqual(props[0]['E'], 21e9)
            self.assertEqual(props[0]['estado'], 'deformacion_plana')

    def test_malla_mixta(self):
        m, props, ns = leer_malla('mallas/ejemplo_mixto.txt')
        self.assertEqual(m['elements']['type'], ['Q4', 'T3', 'T3'])

    def test_errores(self):
        def leer(txt):
            with tempfile.TemporaryDirectory() as d:
                r = os.path.join(d, 'm.txt')
                with open(r, 'w') as f:
                    f.write(txt)
                return leer_malla(r)
        base = "*MATERIAL a plane_stress E=1 nu=0.3\n*NODES\n1 0 0\n2 1 0\n3 0 1\n"
        with self.assertRaises(Exception):
            leer(base + "*ELEMENTS\n1 TRI 1 a 1 2 9\n")        # nodo inexistente
        with self.assertRaises(Exception):
            leer(base + "*ELEMENTS\n1 TRI 1 a 1 2\n")          # faltan nodos
        with self.assertRaises(Exception):
            leer(base + "*ELEMENTS\n1 TRI 1 otro 1 2 3\n")     # material inexistente


class TestFisica(unittest.TestCase):
    def test_peso_propio_total(self):
        for tipo in TIPOS:
            mesh, _ = malla_presa(tipo)
            F = fuerzas_volumen(mesh, [PROP])
            W = 2400 * G * 315.0
            self.assertAlmostEqual(F[1::2].sum() / -W, 1.0, places=10, msg=tipo)
            self.assertAlmostEqual(abs(F[0::2].sum()), 0.0, places=6)

    def test_presion_resultante_y_momento(self):
        H = 28.0
        for tipo in TIPOS:
            mesh, ns = malla_presa(tipo)
            ar = aristas_en_nset(mesh, ns['aguas_arriba'])
            F = fuerzas_presion(mesh, [PROP], ar, presion_hidrostatica(H))
            Fx = F[0::2].sum()
            self.assertAlmostEqual(Fx / (RHO_AGUA * G * H ** 2 / 2), 1.0, places=10, msg=tipo)
            y = mesh['nodes']['position'][:, 1]
            self.assertAlmostEqual((F[0::2] * y).sum() / (RHO_AGUA * G * H ** 3 / 6), 1.0, places=10)
            self.assertAlmostEqual(abs(F[1::2].sum()), 0.0, places=6)

    def test_presion_normal_inclinada(self):
        # paramento inclinado: la fuerza vertical es rho g mu H^2/2 hacia abajo
        p = Presa(mu=0.2)
        mesh, ns = malla_presa('Q8', presa=p)
        ar = aristas_en_nset(mesh, ns['aguas_arriba'])
        F = fuerzas_presion(mesh, [PROP], ar, presion_hidrostatica(28.0))
        self.assertAlmostEqual(F[1::2].sum() / (-RHO_AGUA * G * 0.2 * 28 ** 2 / 2), 1.0, places=10)

    def test_restricciones(self):
        mesh, ns = malla_presa('Q4')
        self.assertEqual(restricciones_base(mesh, ns['base'], 'empotrada').sum(), 2 * len(ns['base']))
        self.assertEqual(restricciones_base(mesh, ns['base'], 'deslizante').sum(), len(ns['base']) + 1)

    def test_robin_suma_rigidez_correcta(self):
        # un resorte kn sobre la base horizontal de largo B: suma de la diagonal vertical = kn * B
        for tipo in TIPOS:
            mesh, ns = malla_presa(tipo)
            ar = aristas_en_nset(mesh, ns['base'])
            for modo in ('diagonal', 'consistente'):
                Kr = resortes_robin(mesh, [PROP], ar, 7.0e9, 3.0e9, modo)
                # la suma de todos los terminos de cada componente es k * longitud (consistente
                # y diagonal reparten distinto, pero conservan la rigidez total)
                self.assertAlmostEqual(Kr[1::2, 1::2].sum() / (7.0e9 * 18.0), 1.0, places=10, msg=(tipo, modo))
                self.assertAlmostEqual(Kr[0::2, 0::2].sum() / (3.0e9 * 18.0), 1.0, places=10)
                if modo == 'diagonal':
                    off = Kr - __import__('scipy.sparse', fromlist=['diags']).diags(Kr.diagonal())
                    self.assertEqual(abs(off).max(), 0.0)

    def test_robin_infinito_equivale_a_empotrada(self):
        for tipo in ('T3', 'Q8'):
            a = analizar_presa(f'mallas/presa_{tipo}.txt')
            b = analizar_presa(f'mallas/presa_{tipo}.txt', bc='robin', kn=1e16)
            self.assertLess(abs(a['D']).max() and abs(a['D'] - b['D']).max() / abs(a['D']).max(), 1e-6)


class TestSolverYPost(unittest.TestCase):
    def test_equilibrio_global_todos_los_tipos(self):
        for tipo in TIPOS:
            r = analizar_presa(f'mallas/presa_{tipo}.txt')
            m = r['momentos']
            self.assertAlmostEqual(m['Rx_tot'] / (-m['empuje']), 1.0, places=9, msg=tipo)
            self.assertAlmostEqual(m['Ry_tot'] / m['peso'], 1.0, places=9)
            self.assertAlmostEqual(m['M_R'] / -(m['M_est'] - m['M_des']), 1.0, places=9)
            self.assertAlmostEqual(m['FS'], 2.4500, places=3)

    def test_momentos_analiticos(self):
        r = analizar_presa('mallas/presa_Q4.txt')
        m = r['momentos']
        self.assertAlmostEqual(m['A'], 315.0, places=9)
        self.assertAlmostEqual(m['M_des'], 1000 * G * 28 ** 3 / 6, places=3)
        self.assertAlmostEqual(m['xG'], 6.142857142857143, places=9)

    def test_solver_barra_simple(self):
        # una barra Q4 de 1x1 traccionada: u = sigma L / E
        mesh = crear_mesh([[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]], 'Q4')
        prop = {'E': 100.0, 'nu': 0.0, 't': 1.0}
        K = ensamblar_K(mesh, [prop])
        F = np.zeros(8)
        F[2] = F[4] = 0.5                                  # traccion total 1 sobre el borde x=1
        fixed = np.zeros(8, bool)
        fixed[[0, 1, 6]] = True                            # nodo 0 empotrado, nodo 3 u=0
        D, R = resolver(K, F, fixed)
        self.assertAlmostEqual(D[2], 0.01, places=12)
        self.assertAlmostEqual(R[0] + R[6], -1.0, places=12)

    def test_extrapolacion_campo_lineal(self):
        # desplazamientos lineales -> tension constante -> nodos exactos para todos los tipos
        for tipo in TIPOS:
            mesh, _ = malla_presa(tipo, 2, 3)
            xy = mesh['nodes']['position']
            D = np.zeros(xy.size)
            D[0::2] = 1e-4 * xy[:, 0] + 2e-5 * xy[:, 1]
            D[1::2] = -3e-5 * xy[:, 0] + 5e-5 * xy[:, 1]
            prop = dict(PROP)
            sig = tensiones_nodales(mesh, [prop], D)
            self.assertLess(np.ptp(sig[:, 0]) + np.ptp(sig[:, 1]) + np.ptp(sig[:, 2]), 1e-3 * abs(sig).max(), tipo)

    def test_principales(self):
        s1, s2, ang = principales(np.array([[100., 0, 0], [0, 0, 50.], [60., -20., 30.]]))
        np.testing.assert_allclose(s1[:2], [100, 50])
        np.testing.assert_allclose(s2[:2], [0, -50])
        self.assertAlmostEqual(s1[2], 20 + np.hypot(40, 30))
        self.assertAlmostEqual(ang[1], np.pi / 4)


if __name__ == '__main__':
    unittest.main()
