"""Tests del ROL 1: elementos.py, ensamble.py y patch_test.py."""
import unittest

import numpy as np
import scipy.sparse as sp

from src.elementos import (NODOS_NAT, N_NODOS, funciones_forma, jacobiano, matriz_B, matriz_D,
                           matriz_Kele, obtener_B_y_D, obtener_N_y_detJ, obtener_puntos_gauss)
from src.ensamble import armar_K_global, crear_mesh, ensamblar_K
import patch_test

TIPOS = ['T3', 'T6', 'Q4', 'Q8', 'Q9']
PROP = {'E': 200e9, 'nu': 0.3, 't': 0.002}


def puntos(tipo):
    rng = np.random.default_rng(0)
    return rng.uniform(0, 0.5, (6, 2)) if tipo[0] == 'T' else rng.uniform(-0.9, 0.9, (6, 2))


def elemento_generico(tipo):
    """Elemento no rectangular con nodos medios en los puntos medios."""
    q = np.array([[0, 0], [3, 0.2], [2.6, 1.5], [0.1, 1.2]])
    t = np.array([[0, 0], [2, 0.3], [0.4, 1.4]])
    base, cod = (t, 'T3') if tipo[0] == 'T' else (q, 'Q4')
    return np.array([funciones_forma(cod, a, b)[0] @ base for a, b in NODOS_NAT[tipo]])


class TestFuncionesForma(unittest.TestCase):
    def test_delta_de_kronecker(self):
        for t in TIPOS:
            for i, (xi, eta) in enumerate(NODOS_NAT[t]):
                np.testing.assert_allclose(funciones_forma(t, xi, eta)[0], np.eye(N_NODOS[t])[i],
                                           atol=1e-14, err_msg=t)

    def test_particion_de_la_unidad_y_derivadas(self):
        h = 1e-6
        for t in TIPOS:
            for xi, eta in puntos(t):
                N, dN = funciones_forma(t, xi, eta)
                self.assertAlmostEqual(N.sum(), 1.0, places=13)
                np.testing.assert_allclose(dN.sum(axis=1), 0, atol=1e-13)
                dxi = (funciones_forma(t, xi + h, eta)[0] - funciones_forma(t, xi - h, eta)[0]) / (2 * h)
                deta = (funciones_forma(t, xi, eta + h)[0] - funciones_forma(t, xi, eta - h)[0]) / (2 * h)
                np.testing.assert_allclose(dN[0], dxi, atol=1e-7)
                np.testing.assert_allclose(dN[1], deta, atol=1e-7)

    def test_isoparametrico_reproduce_coordenadas(self):
        for t in TIPOS:
            for xi, eta in puntos(t):
                N, _ = funciones_forma(t, xi, eta)
                self.assertAlmostEqual(N @ NODOS_NAT[t][:, 0], xi, places=13)
                self.assertAlmostEqual(N @ NODOS_NAT[t][:, 1], eta, places=13)

    def test_tipo_desconocido(self):
        with self.assertRaises(ValueError):
            funciones_forma('Q16', 0, 0)


class TestMaterialYGauss(unittest.TestCase):
    def test_D_tension_plana(self):
        D = matriz_D(200e9, 0.3)
        c = 200e9 / (1 - 0.09)
        np.testing.assert_allclose(D, c * np.array([[1, 0.3, 0], [0.3, 1, 0], [0, 0, 0.35]]))

    def test_D_deformacion_plana(self):
        D = matriz_D(1.0, 0.25, 'deformacion_plana')
        c = 1 / (1.25 * 0.5)
        np.testing.assert_allclose(D, c * np.array([[0.75, 0.25, 0], [0.25, 0.75, 0], [0, 0, 0.25]]))

    def test_pesos_y_exactitud(self):
        esperado = {'T3': 1, 'T6': 3, 'Q4': 4, 'Q8': 9, 'Q9': 9}
        for t, n in esperado.items():
            pts, w = obtener_puntos_gauss(t)
            self.assertEqual((pts.shape, w.shape), ((n, 2), (n,)))
            self.assertAlmostEqual(w.sum(), 0.5 if t[0] == 'T' else 4.0)
        pts, w = obtener_puntos_gauss('Q9')            # 3x3 integra exacto x^2 y^2 (=4/9)
        self.assertAlmostEqual(np.sum(w * pts[:, 0] ** 2 * pts[:, 1] ** 2), 4 / 9)
        pts, w = obtener_puntos_gauss('T6')            # exacta hasta grado 2: int x^2 = 1/12
        self.assertAlmostEqual(np.sum(w * pts[:, 0] ** 2), 1 / 12)
        self.assertEqual(len(obtener_puntos_gauss('Q8', 'tensiones')[1]), 4)


class TestRigidezElemento(unittest.TestCase):
    def test_T3_contra_formula_de_clase(self):
        xy = np.array([[0.0, 0.0], [2.0, 0.3], [0.4, 1.4]])
        (x1, y1), (x2, y2), (x3, y3) = xy
        detA = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)          # = 2 * area
        B = np.array([[y2 - y3, 0, y3 - y1, 0, y1 - y2, 0],
                      [0, x3 - x2, 0, x1 - x3, 0, x2 - x1],
                      [x3 - x2, y2 - y3, x1 - x3, y3 - y1, x2 - x1, y1 - y2]]) / detA
        Ke = PROP['t'] * (detA / 2) * B.T @ matriz_D(PROP['E'], PROP['nu']) @ B
        np.testing.assert_allclose(matriz_Kele(xy, 'T3', PROP), Ke, rtol=1e-12)

    def test_simetria_y_modos_de_cuerpo_rigido(self):
        for t in TIPOS:
            xy = elemento_generico(t)
            K = matriz_Kele(xy, t, PROP)
            np.testing.assert_allclose(K, K.T, atol=1e-9 * abs(K).max())
            w = np.linalg.eigvalsh(K)
            self.assertEqual(int(np.sum(w < 1e-9 * w.max())), 3, t)
            trasl = np.tile([1.0, 0.0], len(xy))
            rot = np.column_stack([-xy[:, 1], xy[:, 0]]).ravel()
            for modo in (trasl, rot):
                self.assertLess(abs(K @ modo).max(), 1e-8 * abs(K).max())

    def test_area_con_detJ(self):
        xy = np.array([[0, 0], [4, 0], [4, 2], [0, 2.0]])
        pts, w = obtener_puntos_gauss('Q4')
        area = sum(wi * obtener_N_y_detJ(xy, 'Q4', a, b)[1] for (a, b), wi in zip(pts, w))
        self.assertAlmostEqual(area, 8.0)

    def test_detJ_negativo(self):
        with self.assertRaises(ValueError):
            matriz_Kele(np.array([[0, 0], [0, 1], [1, 1], [1, 0.0]]), 'Q4', PROP)


class TestEnsamble(unittest.TestCase):
    def malla_mixta(self):
        nodos = [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]]
        conn = [[0, 1, 2, 3], [1, 4, 5], [1, 5, 2]]
        return nodos, conn, ['Q4', 'T3', 'T3']

    def test_K_global_mixta(self):
        nodos, conn, tipos = self.malla_mixta()
        mesh = crear_mesh(nodos, conn, tipos)
        K = ensamblar_K(mesh, [PROP])
        self.assertTrue(sp.isspmatrix_csr(K))
        self.assertEqual(K.shape, (12, 12))
        self.assertLess(abs(K - K.T).max(), 1e-9 * abs(K).max())
        pos = np.array(nodos, float)
        for modo in (np.tile([1.0, 0.0], 6), np.tile([0.0, 1.0], 6),
                     np.column_stack([-pos[:, 1], pos[:, 0]]).ravel()):
            self.assertLess(abs(K @ modo).max(), 1e-8 * abs(K).max())
        self.assertEqual(int(np.sum(np.linalg.eigvalsh(K.toarray()) < 1e-9 * abs(K).max())), 3)

    def test_ensamble_suma_contribuciones(self):
        # dos Q4 que comparten una arista: K[nodo compartido] = suma de ambos elementos
        nodos = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
        K = ensamblar_K(crear_mesh(nodos, [[0, 1, 4, 3], [1, 2, 5, 4]], 'Q4'), [PROP]).toarray()
        K1 = matriz_Kele(np.array(nodos)[[0, 1, 4, 3]], 'Q4', PROP)
        K2 = matriz_Kele(np.array(nodos)[[1, 2, 5, 4]], 'Q4', PROP)
        np.testing.assert_allclose(K[2:4, 2:4], K1[2:4, 2:4] + K2[0:2, 0:2], rtol=1e-12)

    def test_contrato_armar_K_global(self):
        nodos, conn, tipos = self.malla_mixta()
        K1 = armar_K_global(nodos[:4], [conn[0]], PROP, 'Q4')
        self.assertEqual(K1.shape, (8, 8))
        K2 = ensamblar_K(crear_mesh(nodos, conn, tipos), [PROP])
        K3 = armar_K_global(nodos, conn, [PROP], tipos)
        self.assertLess(abs(K2 - K3).max(), 1e-12 * abs(K2).max())

    def test_contrato_N_B_D(self):
        xy = elemento_generico('Q8')
        N, detJ = obtener_N_y_detJ(xy, 'Q8', 0.2, -0.3)
        self.assertEqual(N.shape, (8,))
        self.assertGreater(detJ, 0)
        B, D = obtener_B_y_D(xy, PROP, 'Q8', 0.2, -0.3)
        self.assertEqual((B.shape, D.shape), ((3, 16), (3, 3)))
        # un campo lineal u = a + b x + c y da deformaciones constantes iguales a b, c
        de = np.zeros(16)
        de[0::2] = 1e-3 * xy[:, 0] + 2e-3 * xy[:, 1]
        de[1::2] = -5e-4 * xy[:, 1]
        np.testing.assert_allclose(B @ de, [1e-3, -5e-4, 2e-3], atol=1e-15)

    def test_errores_de_entrada(self):
        with self.assertRaises(ValueError):
            crear_mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], 'Q4')
        with self.assertRaises(ValueError):
            crear_mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 5]], 'T3')


class TestPatch(unittest.TestCase):
    def test_patch_test_todos(self):
        for tipo in TIPOS:
            for variante in ('cargas', 'desplazamientos'):
                for estado in patch_test.ESTADOS:
                    for semilla in (1, 3):
                        with self.subTest(tipo=tipo, variante=variante, estado=estado, semilla=semilla):
                            es, eu = patch_test.correr_patch(tipo, estado, variante, semilla=semilla)
                            self.assertLess(es, patch_test.TOL)
                            self.assertLess(eu, patch_test.TOL)


if __name__ == '__main__':
    unittest.main()
