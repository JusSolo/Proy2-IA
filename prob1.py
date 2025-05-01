
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import imageio

def guardar_gif(frames, nombre="dfs_laberinto.gif", duracion=0.05, escala=5):
    """
    Guarda un GIF ampliando la resolución por un factor 'escala'.

    :param frames: lista de imágenes (arrays numpy RGB)
    :param nombre: nombre del archivo GIF a guardar
    :param duracion: duración de cada frame en segundos
    :param escala: factor de ampliación (por ejemplo 5 para quintuplicar)
    """
    frames_grandes = [zoom(frame, (escala, escala, 1), order=0) for frame in frames]
    imageio.mimsave(nombre, frames_grandes, duration=duracion)


def find(a, L):
    for i in range(len(L)):
        S = L[i]
        if a in S:
            return i


def notConnected(A):
    from collections import deque

    # Obtener todos los vértices
    vertices = set()
    for u, v in A:
        vertices.add(u)
        vertices.add(v)

    if not vertices:
        return True

    # Crear lista de adyacencia
    adj = {v: [] for v in vertices}
    for u, v in A:
        adj[u].append(v)
        adj[v].append(u)

    # Hacer BFS desde un vértice cualquiera
    start = next(iter(vertices))
    visited = set([start])
    queue = deque([start])
    while queue:
        curr = queue.popleft()
        for neighbor in adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # Si todos los vértices fueron visitados, está conectado
    return visited != vertices

from collections import deque

def is_connected(edges, vertices):
    """Devuelve True si el grafo (vertices, edges) es conexo."""
    # Construyo la lista de adyacencia incluyendo vértices aislados
    adj = {v: [] for v in vertices}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # BFS
    start = next(iter(vertices))
    visited = {start}
    queue = deque([start])
    while queue:
        curr = queue.popleft()
        for w in adj[curr]:
            if w not in visited:
                visited.add(w)
                queue.append(w)
    return len(visited) == len(vertices)

def MazeKruskals(M,N,seed = 313):
    frames = []
    Maze = np.zeros((2*M-1,2*N-1),int)
    # Conjunto de aristas todas con el mismo peso
    A = []
    for i in range(1,M):
        for j in range(1,N):
            A.append(((i-1,j),(i,j)))
            A.append(((i,j-1),(i,j)))
        A.append(((i,0),(i-1,0)))
    for j in range(1,N):
         A.append(((0,j),(0,j-1)))
    random.seed(seed)
    A = sorted(A, key=lambda x: random.random())

    # V lista de vertices
    V = [ (i,j) for i in range(M) for j in range(N) ]
    # PV particion de los vertices (en forma de lista)
    PV = [frozenset({v}) for v in V]
    T = []
    for a in A:
        u , v = a

        i , j = find(u,PV) , find(v,PV)

        if i != j:

            T.append(a)
            PV[j] = PV[j].union(PV[i])
            PV.pop(i)

            # pintar el laberinto Corrige esta parte
            Maze[tuple(2*np.array(u))] = 1
            Maze[tuple(2*np.array(v))] = 1
            Maze[tuple(np.array(u) + np.array(v))] = 1
            frames.append(Maze.copy())
    return T,Maze,frames





def MazeReverseDelete(M, N, seed=313):
    frames = []
    # laberinto inicial
    Maze = np.ones((2*M-1,2*N-1),int)
    for i in range(1,M*2-1,2):
        for j in range(1,N*2-1,2):
            Maze[i,j] = 0
    frames.append(Maze.copy())
    # Conjunto de aristas todas con el mismo peso
    A = []
    for i in range(1,M):
        for j in range(1,N):
            A.append(((i-1,j),(i,j)))
            A.append(((i,j-1),(i,j)))
        A.append(((i,0),(i-1,0)))
    for j in range(1,N):
         A.append(((0,j),(0,j-1)))
    random.seed(seed)
    A = sorted(A, key=lambda x: random.random())
    V = [ (i,j) for i in range(M) for j in range(N) ]
    i = 0
    while i < len(A):
        Aa = A.copy()
        a = A.pop(i)
        if not is_connected(A,V):#if notConnected(A):
            A = Aa
            i += 1
        else:
             u , v = a
             Maze[tuple(np.array(u) + np.array(v))] = 0
             frames.append(Maze.copy())
    return A,Maze,frames
