from prob1 import MazeKruskals  # Importar la función
import matplotlib.pyplot as plt
import numpy as np

def visitables(Maze, nodoa, nodop):
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

def DFS(Maze, inicio=(0, 0), fin=(60, 80)):
    frames = []


    nodo = inicio
    nodop = (0, -1)
    camino = [nodo]
    rep = False
    f = np.stack([Maze * 255] * 3, axis=-1).astype(np.uint8)
    f[nodo] = [243, 255, 141]
    frames.append(f.copy())



    while nodo != fin:
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
            if camino[-1] == nodo:
                camino.pop(-1)
                f[nodo] = [180, 180, 180]
            else:
                if rep:
                    f[nodop] = [243, 255, 141] #[66, 254, 161] (verde)
                    camino.append(nodop)
                    rep = False
                f[nodo] = [243, 255, 141] #[66, 254, 161]
                camino.append(nodo)
        frames.append(f)


    return camino, frames

# ejemplo:

m, n = 4, 8
_, M, _ = MazeKruskals(m, n, seed=313)
camino, frames = DFS(M, inicio=(0, 0), fin=(2*m - 2, 2*n - 2))  # Fin = esquina opuesta

# Mostrar último frame
print(camino)
plt.imshow(frames[-1])

plt.show()
