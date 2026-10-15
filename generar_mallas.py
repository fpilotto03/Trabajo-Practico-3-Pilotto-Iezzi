"""Genera las mallas de la presa en texto plano (formato del parser) en mallas/*.txt."""
import os
import numpy as np
from src.geometria import Presa, generar_malla_presa
from src.ensamble import crear_mesh
from src.parser import escribir_malla, leer_malla, verificar_orientacion

PROP = {'nombre': 'hormigon', 'estado': 'deformacion_plana', 'E': 21e9, 'nu': 0.25, 'rho': 2400.0, 't': 1.0}


def escribir(ruta, presa, tipo, nx, ny, titulo):
    nodos, conn, nsets = generar_malla_presa(presa, tipo, nx, ny)
    mesh = crear_mesh(nodos, conn, tipo)
    verificar_orientacion(mesh)
    escribir_malla(ruta, mesh, [PROP], nsets, titulo)


if __name__ == '__main__':
    os.makedirs('mallas', exist_ok=True)
    presa = Presa()
    for tipo, (nx, ny) in {'T3': (8, 30), 'T6': (6, 24), 'Q4': (8, 30), 'Q8': (6, 24), 'Q9': (6, 24)}.items():
        ruta = f'mallas/presa_{tipo}.txt'
        escribir(ruta, presa, tipo, nx, ny, f'Presa de gravedad H=30 m, base 18 m - {tipo}')
        print('escrita', ruta)
    # malla con elementos mixtos (Q4 + T3) para mostrar el formato
    nodos = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], float)
    conn = [[0, 1, 4, 3], [1, 2, 5], [1, 5, 4]]
    mesh = crear_mesh(nodos, conn, ['Q4', 'T3', 'T3'])
    escribir_malla('mallas/ejemplo_mixto.txt', mesh, [PROP],
                   {'izquierda': np.array([0, 3])}, 'Ejemplo de malla mixta Q4 + T3')
    leer_malla('mallas/ejemplo_mixto.txt')
    print('escrita mallas/ejemplo_mixto.txt')
