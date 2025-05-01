from prob1 import MazeKruskals, guardar_gif  # Importar la función
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import imageio



def visitables1(Maze, nodoa, nodop):
    L, C = Maze.shape

    T = np.array([[ 0, 1],
                  [-1, 0]])

    nodop = np.array(nodop)
    nodoa = np.array(nodoa)
    f2 = nodoa - nodop
    f1 = f2 @ T
    f3 = f1 @ T @ T
    v = []
    for f in [f1, f2, f3]:
        x, y = tuple(f + nodoa)
        if 0 <= x < L and 0 <= y < C and Maze[x, y] == 1:
            v.append((x, y))
    return v

def visitables2(Maze, nodoa):
    L, C = Maze.shape

    nodoa = np.array(nodoa)
    f1 = nodoa + np.array((0,1))
    f2 = nodoa + np.array((1,0))
    f3 = nodoa + np.array((0,-1))
    f4 = nodoa + np.array((-1,0))
    v = []
    for f in [f1, f2, f3,f4]:
        x, y = tuple(f)
        if 0 <= x < L and 0 <= y < C and Maze[x, y] == 1:
            v.append((x, y))
    return v

def visitables(Maze, nodoa, nodop=None):
    if nodop is None:
        return visitables2(Maze, nodoa)
    else:
        return visitables1(Maze, nodoa, nodop)

#####################################################
# Busqueda en profundidad
#####################################################

def DFS(Maze, inicio=(0, 0), fin=(60, 80)):
    frames = []


    nodo = inicio
    nodop = (0, -1)
    camino = [nodo]
    rep = False
    f = np.stack([Maze * 255] * 3, axis=-1).astype(np.uint8)
    f[nodo] = [243, 255, 141]
    frames.append(f.copy())

    nodosExplorados = 1
    pasos = 1
    while nodo != fin:
        pasos +=1
        f = frames[-1].copy()

        posibles = visitables(Maze, nodo, nodop)

        if len(posibles) == 0:
            # Retroceso (gris)
            f[nodo] = [180, 180, 180]
            nodo, nodop = nodop, nodo
            f[nodo] = [180, 180, 180]
            camino.pop(-1)
            camino.pop(-1)
            plt.imshow(f)

            rep = True
        else:
            #camino dorado
            nodo, nodop = posibles[0], nodo
            if camino[-1] == nodo: # retroceso
                camino.pop(-1)
                f[nodo] = [180, 180, 180]
            else: #nuevo nodo explorado
                nodosExplorados +=1
                if rep:
                    f[nodop] = [243, 255, 141] #[66, 254, 161] (verde)
                    camino.append(nodop)
                    rep = False
                f[nodo] = [243, 255, 141] #[66, 254, 161]
                camino.append(nodo)
        frames.append(f)


    return camino, frames, pasos , nodosExplorados

############################################################
# Busqueda en Anchura
########################################################


def BFS(Maze, inicio, fin):
    frames = []
    cola = {}
    padre = {} #diccionario del padre de cada nodo
    nodo = inicio
    cola = deque([inicio])
    visitados = set([inicio])
    #nodop = (-1,0)
    nodo = (-1,0)
    pasos = 0

    # Imagen inicial
    f = np.stack([Maze * 255] * 3, axis=-1).astype(np.uint8)
    f[inicio] = [180, 180, 180] #[243, 255, 141] #[66, 254, 161]
    frames.append(f.copy())
    while cola:
        pasos +=1
        #nodop = nodo
        nodo = cola.popleft()
        if nodo == fin:
            break

        for v in visitables(Maze, nodo):
            if v not in visitados:
                cola.append(v)
                visitados.add(v)
                padre[v] = nodo
                f[v] = [180, 180, 180] #[66, 254, 161]
        frames.append(f.copy())
    # encontrar finalemente el camino
    camino = []
    while nodo != inicio:
        camino.append(nodo)
        f[nodo] = [243, 255, 141]
        frames.append(f.copy())
        nodo = padre[nodo]
    camino.append(inicio)
    f[inicio] = [243, 255, 141]
    frames.append(f.copy())
    camino.reverse()
    nodosExplorados = len(visitados)

    return camino, frames, pasos, nodosExplorados


# ejemplo:

m, n = 32, 42 # usar siempre numeros pares
_, M, _ = MazeKruskals(m, n, seed=313)
camino, frames, ex,i = BFS(M, inicio=(0, 0), fin=(2*m - 2, 2*n - 2))  # Fin = esquina opuesta

# Mostrar último frame
print(camino)
print(f"nodos visitados: {ex}")
print(f"iterciones: {i}")
plt.imshow(frames[-1])

plt.show()
guardar_gif(frames, "dfs_laberinto.gif", duracion=0.05)
