# 📌 Proyecto 2 - Inteligencia Artificial

**Universidad del Valle de Guatemala**  
**Curso: Inteligencia Artificial**  
**Segundo Proyecto – Algoritmos de Búsqueda en Laberintos**  
Autores: Gabriel Alberto Paz (221087), Juan Luis Solórzano (201598), Hansel Andrés López (19026)

---

## 📘 Descripción

Este proyecto implementa y compara distintos algoritmos de búsqueda para resolver laberintos generados aleatoriamente. Se divide en tres partes:

1. **Generación de laberintos** mediante algoritmos de árbol de expansión mínima.
2. **Resolución visual** de laberintos usando BFS y DFS.
3. **Comparación experimental** entre cuatro algoritmos de búsqueda: BFS, DFS, UCS y A*.

---

## 🧩 Problemas resueltos

### 🧱 Problema 1: Generación de Laberintos

- Algoritmos utilizados:
  - Kruskal
  - Reverse-delete
- Visualización incluida: `dfs_laberinto.gif`

### 🧭 Problema 2: Solución de Laberintos

- Punto de inicio: esquina superior izquierda
- Punto final: esquina inferior derecha
- Algoritmos implementados:
  - Búsqueda en Profundidad (DFS)
  - Búsqueda en Anchura (BFS)
- Resultado visual generado: `laberinto_resuelto.png`

### 📊 Problema 3: Comparación de Algoritmos

- Algoritmos comparados:
  - BFS
  - DFS
  - UCS (Dijkstra)
  - A*
- Se ejecutaron **25 simulaciones aleatorias**
- Resultados guardados en `results_prob3.csv`
- Gráfica generada: `comparacion_algoritmos.png`

---

## 🚀 Cómo ejecutar

1. Clona el repositorio:
   ```bash
    git clone https://github.com/JusSolo/Proy2-IA
    cd Proy2-IA
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python prob1.py
    python prob2.py
    python prob3.py
   ```