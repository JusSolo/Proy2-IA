import random
import time
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque
from prob1 import MazeKruskals

# ➡️ añadir colorama
from colorama import init, Fore, Style

init(autoreset=True)

# —————————————————————————————————————————————
# ——— Funciones de búsqueda ————————
# —————————————————————————————————————————————

def get_neighbors(maze, node):
    L, C = maze.shape
    x, y = node
    for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < L and 0 <= ny < C and maze[nx, ny] == 1:
            yield (nx, ny)

def bfs(maze, start, goal):
    t0 = time.perf_counter()
    queue = deque([start]); visited = {start}; parent = {}; nodes = 0
    while queue:
        curr = queue.popleft()
        nodes += 1
        if curr == goal: break
        for nb in get_neighbors(maze, curr):
            if nb not in visited:
                visited.add(nb); parent[nb] = curr; queue.append(nb)
    path, node = [], goal
    while node != start:
        path.append(node); node = parent[node]
    path.append(start); path.reverse()
    return {'nodes': nodes, 'length': len(path)-1, 'time': time.perf_counter() - t0}

def dfs(maze, start, goal):
    t0 = time.perf_counter()
    stack = [start]; visited = set(); parent = {}; nodes = 0
    while stack:
        curr = stack.pop()
        if curr in visited: continue
        visited.add(curr); nodes += 1
        if curr == goal: break
        for nb in get_neighbors(maze, curr):
            if nb not in visited:
                parent[nb] = curr; stack.append(nb)
    path, node = [], goal
    while node != start:
        path.append(node); node = parent[node]
    path.append(start); path.reverse()
    return {'nodes': nodes, 'length': len(path)-1, 'time': time.perf_counter() - t0}

def ucs(maze, start, goal):
    t0 = time.perf_counter()
    heap = [(0, start)]; cost = {start:0}; parent = {}; visited = set(); nodes = 0
    while heap:
        c, curr = heapq.heappop(heap)
        if curr in visited: continue
        visited.add(curr); nodes += 1
        if curr == goal: break
        for nb in get_neighbors(maze, curr):
            nc = c+1
            if nb not in cost or nc < cost[nb]:
                cost[nb] = nc; parent[nb] = curr
                heapq.heappush(heap, (nc, nb))
    path, node = [], goal
    while node != start:
        path.append(node); node = parent[node]
    path.append(start); path.reverse()
    return {'nodes': nodes, 'length': len(path)-1, 'time': time.perf_counter() - t0}

def astar(maze, start, goal):
    t0 = time.perf_counter()
    def h(u,v): return abs(u[0]-v[0]) + abs(u[1]-v[1])
    heap = [(h(start, goal), 0, start)]
    cost = {start:0}; parent = {}; visited = set(); nodes = 0
    while heap:
        f, g, curr = heapq.heappop(heap)
        if curr in visited: continue
        visited.add(curr); nodes += 1
        if curr == goal: break
        for nb in get_neighbors(maze, curr):
            ng = g+1
            if nb not in cost or ng < cost[nb]:
                cost[nb] = ng; parent[nb] = curr
                heapq.heappush(heap, (ng + h(nb, goal), ng, nb))
    path, node = [], goal
    while node != start:
        path.append(node); node = parent[node]
    path.append(start); path.reverse()
    return {'nodes': nodes, 'length': len(path)-1, 'time': time.perf_counter() - t0}

# —————————————————————————————————————————————
# ——— Loop de experimentos ——————————————————
# —————————————————————————————————————————————

def run_experiments(K=25, M=45, N=55):
    random.seed(1234)
    algs = {'BFS': bfs, 'DFS': dfs, 'UCS': ucs, 'A*': astar}
    records = []

    for i in range(1, K+1):
        print(f"{Fore.CYAN}🔀 Escenario {i}/{K}...{Style.RESET_ALL}")
        _, maze, _ = MazeKruskals(M, N, seed=random.randint(0,10**6))

        # elegir start/goal con Manhattan >=10
        while True:
            u1, v1 = random.randrange(M), random.randrange(N)
            u2, v2 = random.randrange(M), random.randrange(N)
            if abs(u1-u2) + abs(v1-v2) >= 10:
                start = (2*u1, 2*v1); goal = (2*u2, 2*v2)
                break

        metrics = {}
        for name, func in algs.items():
            print(f"  {Fore.YELLOW}➡️  Ejecutando {name}...{Style.RESET_ALL}", end="")
            res = func(maze, start, goal)
            metrics[name] = res
            # mostramos un pequeño resumen inmediato
            print(f"  {Fore.GREEN}✔ nodos={res['nodes']}, len={res['length']}, t={res['time']:.4f}s{Style.RESET_ALL}")

        order = sorted(algs.keys(), key=lambda n: metrics[n]['nodes'])
        rank  = {n: r+1 for r,n in enumerate(order)}

        rec = {'Escenario': i, 'Start': start, 'Goal': goal}
        for name in algs:
            rec[f'{name}_nodes'] = metrics[name]['nodes']
            rec[f'{name}_len']   = metrics[name]['length']
            rec[f'{name}_time']  = metrics[name]['time']
            rec[f'{name}_rank']  = rank[name]
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv('results_prob3.csv', index=False)
    return df

if __name__ == '__main__':
    print(f"{Fore.MAGENTA}{Style.BRIGHT}🚀 Iniciando Experimento Problema 3{Style.RESET_ALL}\n")
    df = run_experiments()

    # Mostrar tabla completa con colores
    print(f"\n{Fore.BLUE}{Style.BRIGHT}📋 Resultados completos:{Style.RESET_ALL}")
    print(df.to_string(index=False))

    # 1) Calcula el promedio de nodos explorados para cada algoritmo
    avg_nodes = df[['BFS_nodes','DFS_nodes','UCS_nodes','A*_nodes']].mean()
    # 2) Ordena de menor a mayor
    avg_nodes = avg_nodes.sort_values()

    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}🏆 Ranking por Nodos Explorados Promedio (menor = mejor){Style.RESET_ALL}")
    for i, (col, val) in enumerate(avg_nodes.items(), start=1):
        name = col.replace('_nodes','')    # quita sufijo
        # asigna medalla según posición
        emoji = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else "🎖"
        print(f"{emoji} {Fore.CYAN}{name:<3}{Style.RESET_ALL} → {Fore.YELLOW}{val:.1f}{Style.RESET_ALL} nodos")

    # Gráfica
    avg_nodes = df[['BFS_nodes','DFS_nodes','UCS_nodes','A*_nodes']].mean()
    plt.figure()
    avg_nodes.plot(kind='bar')
    plt.title('📊 Promedio de nodos explorados por algoritmo')
    plt.xlabel('Algoritmo')
    plt.ylabel('Nodos explorados promedio')
    plt.tight_layout()
    plt.show()
