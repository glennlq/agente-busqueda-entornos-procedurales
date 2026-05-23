"""
Marco de Benchmarking Automatizado - Explorador de Mapas IA
Fase 1: Demostración Visual del Caso Base (Comparativa UCS vs A*)
Fase 2: Simulación de Montecarlo y Análisis de Estrés
Desarrollado por:
- Cáceres Flores, Abimael Fernando
- Luque Canaza, Glenn Rozier
- Perez Aguinaga, Jheyder Whitman
- Romero Lovera, Junio Alberto
Profesor: 
- Yvan Jesus Tupac Valdivia
"""
import random
import time
import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any

# --- CONFIGURACIÓN DE INVESTIGACIÓN ---
FILAS: int = 100
COLUMNAS: int = 100
ITERACIONES_SIMULACION: int = 500 
# --------------------------------------

LLANURA: int = 0
BOSQUE: int = 1
OBSTACULO: int = 2
COSTOS: Dict[int, int] = {LLANURA: 1, BOSQUE: 3}
MOVIMIENTOS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def generar_mapa(filas: int, columnas: int) -> List[List[int]]:
    mapa = []
    for _ in range(filas):
        fila = []
        for _ in range(columnas):
            prob = random.random()
            if prob < 0.7: fila.append(LLANURA)
            elif prob < 0.9: fila.append(BOSQUE)
            else: fila.append(OBSTACULO)
        mapa.append(fila)
    return mapa

def es_valido(mapa: List[List[int]], fila: int, columna: int) -> bool:
    if not (0 <= fila < len(mapa) and 0 <= columna < len(mapa[0])): return False
    return mapa[fila][columna] != OBSTACULO

def heuristica_parametrizada(actual: Tuple[int, int], meta: Tuple[int, int], multiplicador: float) -> float:
    distancia = abs(actual[0] - meta[0]) + abs(actual[1] - meta[1])
    return distancia * multiplicador

def reconstruir_camino(padres: Dict[Tuple[int, int], Optional[Tuple[int, int]]], meta: Tuple[int, int]) -> List[Tuple[int, int]]:
    camino, nodo = [], meta
    while nodo is not None:
        camino.append(nodo)
        nodo = padres[nodo]
    camino.reverse()
    return camino

def ucs(mapa: List[List[int]], inicio: Tuple[int, int], meta: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    cola = [(0, inicio)]
    visitados = set()
    padres = {inicio: None}
    costo_acumulado = {inicio: 0}
    nodos_expandidos = 0
    while cola:
        costo_actual, actual = heapq.heappop(cola)
        if actual in visitados: continue
        visitados.add(actual)
        nodos_expandidos += 1
        if actual == meta:
            return {"camino": reconstruir_camino(padres, meta), "costo": costo_actual, "nodos": nodos_expandidos}
        fila, columna = actual
        for df, dc in MOVIMIENTOS:
            nf, nc = fila + df, columna + dc
            if es_valido(mapa, nf, nc):
                vecino = (nf, nc)
                nuevo_costo = costo_actual + COSTOS[mapa[nf][nc]]
                if vecino not in costo_acumulado or nuevo_costo < costo_acumulado[vecino]:
                    costo_acumulado[vecino] = nuevo_costo
                    padres[vecino] = actual
                    heapq.heappush(cola, (nuevo_costo, vecino))
    return None

def astar(mapa: List[List[int]], inicio: Tuple[int, int], meta: Tuple[int, int], multiplicador: float) -> Optional[Dict[str, Any]]:
    cola = [(0, inicio)]
    padres = {inicio: None}
    g = {inicio: 0}
    nodos_expandidos = 0
    visitados = set()
    while cola:
        _, actual = heapq.heappop(cola)
        if actual in visitados: continue
        visitados.add(actual)
        nodos_expandidos += 1
        if actual == meta:
            return {"camino": reconstruir_camino(padres, meta), "costo": g[actual], "nodos": nodos_expandidos}
        fila, columna = actual
        for df, dc in MOVIMIENTOS:
            nf, nc = fila + df, columna + dc
            if es_valido(mapa, nf, nc):
                vecino = (nf, nc)
                nuevo_g = g[actual] + COSTOS[mapa[nf][nc]]
                if vecino not in g or nuevo_g < g[vecino]:
                    g[vecino] = nuevo_g
                    f = nuevo_g + heuristica_parametrizada(vecino, meta, multiplicador)
                    padres[vecino] = actual
                    heapq.heappush(cola, (f, vecino))
    return None

def mostrar_mapa_comparativo(mapa: List[List[int]], inicio: Tuple[int, int], meta: Tuple[int, int], res_ucs: Dict[str, Any], res_astar: Dict[str, Any]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Se ajusta la posición y el peso del título principal
    fig.suptitle('Demostración de Rutas: UCS vs A* (Heurística h=1.3)', fontsize=16, fontweight='bold')
    
    cmap = mcolors.ListedColormap(['#A8E6CF', '#2E7D32', '#424242'])
    
    leyenda_terrenos = [
        mpatches.Patch(color='#A8E6CF', label='Llanura (Costo 1)'),
        mpatches.Patch(color='#2E7D32', label='Bosque (Costo 3)'),
        mpatches.Patch(color='#424242', label='Obstáculo (Infranqueable)'),
        plt.Line2D([0], [0], color='red', lw=2.5, label='Ruta Elegida')
    ]

    for ax, res, titulo in zip([ax1, ax2], [res_ucs, res_astar], ["Búsqueda de Costo Uniforme (UCS)", "A* (Esperanza Matemática h=1.3)"]):
        ax.imshow(mapa, cmap=cmap)
        
        # Se agrega un pequeño padding interno a los subtítulos
        ax.set_title(f"{titulo}\nNodos: {res['nodos']} | Costo: {res['costo']}", fontsize=12, pad=10)
        
        ax.scatter(inicio[1], inicio[0], color='blue', s=120, edgecolors='white', label='Inicio', zorder=5)
        ax.scatter(meta[1], meta[0], color='yellow', s=120, edgecolors='black', label='Meta', zorder=5)
        
        if res and "camino" in res:
            y = [p[0] for p in res["camino"]]
            x = [p[1] for p in res["camino"]]
            ax.plot(x, y, color='red', linewidth=2.5, zorder=4)
            
        ax.legend(handles=leyenda_terrenos, loc='upper right', fontsize=9, facecolor='white', framealpha=0.9)
        ax.axis('off')

    # El parámetro rect=[left, bottom, right, top] limita el área de las subgráficas
    # dejando el 7% superior completamente libre para el suptitle.
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

def ejecutar_demostracion_unitaria():
    print("\n" + "="*105)
    print("NIVEL 1: RESULTADOS DEL MAPA UNITARIO (CASO BASE)")
    print("="*105)
    
    inicio, meta = (0, 0), (99, 99)
    mapa = generar_mapa(FILAS, COLUMNAS)
    mapa[inicio[0]][inicio[1]] = LLANURA
    mapa[meta[0]][meta[1]] = LLANURA

    t0 = time.time()
    res_ucs = ucs(mapa, inicio, meta)
    t_ucs = (time.time() - t0) * 1000

    t0 = time.time()
    res_astar = astar(mapa, inicio, meta, 1.3)
    t_astar = (time.time() - t0) * 1000

    if not res_ucs or not res_astar:
        print("El mapa generado no tiene solución (encerrado por obstáculos). Reiniciando...")
        return ejecutar_demostracion_unitaria()

    speedup = res_ucs['nodos'] / res_astar['nodos']
    error = ((res_astar['costo'] - res_ucs['costo']) / res_ucs['costo']) * 100

    print(f"{'Algoritmo':<16} | {'Nodos Expandidos':<20} | {'Tiempo (ms)':<14} | {'Costo Total':<13} | {'Rendimiento':<13} | {'Margen Error':<12}")
    print("-" * 105)
    print(f"{'UCS (Baseline)':<16} | {res_ucs['nodos']:<20} | {t_ucs:<14.2f} | {res_ucs['costo']:<13} | {'1.00x':<13} | {'0.00%':<12}")
    print(f"{'A* (h=1.3)':<16} | {res_astar['nodos']:<20} | {t_astar:<14.2f} | {res_astar['costo']:<13} | {f'{speedup:.2f}x':<13} | {f'{error:.2f}%':<12}")
    print("-" * 105)
    print("\nDesplegando gráficos... (Cierra la ventana gráfica para continuar)")
    
    mostrar_mapa_comparativo(mapa, inicio, meta, res_ucs, res_astar)
    input("\n[Presiona Enter para iniciar el Nivel 2: Prueba de Estrés y Benchmarking...]")
    ejecutar_pipeline_investigacion(iteraciones=ITERACIONES_SIMULACION)

def ejecutar_pipeline_investigacion(iteraciones: int):
    multiplicadores = [0.5, 1.0, 1.3, 1.5, 2.0]
    etiquetas_astar = [f"A* (h={m})" for m in multiplicadores]
    
    resultados = {"UCS": {"nodos": [], "tiempo": [], "costo": []}}
    for etiqueta in etiquetas_astar:
        resultados[etiqueta] = {"nodos": [], "tiempo": [], "costo": []}
    
    inicio, meta = (0, 0), (99, 99)
    mapas_procesados = 0
    
    print("\n" + "="*80)
    print("NIVEL 2 - FASE 1: SIMULACIÓN DE MONTECARLO (RECOLECCIÓN DE DATOS)")
    print("="*80)
    print(f"Generando y resolviendo {iteraciones} mapas procedurales únicos...")
    
    while mapas_procesados < iteraciones:
        mapa = generar_mapa(FILAS, COLUMNAS)
        mapa[inicio[0]][inicio[1]] = LLANURA
        mapa[meta[0]][meta[1]] = LLANURA
        
        t0 = time.time()
        res_ucs = ucs(mapa, inicio, meta)
        t_ucs = (time.time() - t0) * 1000
        
        if not res_ucs: continue 
        
        resultados["UCS"]["nodos"].append(res_ucs["nodos"])
        resultados["UCS"]["tiempo"].append(t_ucs)
        resultados["UCS"]["costo"].append(res_ucs["costo"])
        
        for m, etiqueta in zip(multiplicadores, etiquetas_astar):
            t0 = time.time()
            res_astar = astar(mapa, inicio, meta, m)
            t_astar = (time.time() - t0) * 1000
            
            resultados[etiqueta]["nodos"].append(res_astar["nodos"])
            resultados[etiqueta]["tiempo"].append(t_astar)
            resultados[etiqueta]["costo"].append(res_astar["costo"])
            
        mapas_procesados += 1
        if mapas_procesados % 25 == 0:
            print(f"Progreso: {mapas_procesados}/{iteraciones} mapas simulados.")
            
    print("-> Simulación completada con éxito.")
    input("\n[Presiona Enter para pasar a la Fase 2 y 3: Análisis y Benchmarking...]")
    mostrar_analisis_benchmarking(resultados, etiquetas_astar)

def mostrar_analisis_benchmarking(resultados: Dict[str, Dict[str, List[float]]], etiquetas_astar: List[str]):
    print("\n" + "="*80)
    print("NIVEL 2 - FASE 2 Y 3: ANÁLISIS DE SENSIBILIDAD Y BENCHMARKING RIGUROSO")
    print("="*80)
    print(f"{'Algoritmo':<16} | {'Nodos Promedio':<15} | {'Tiempo Prom. (ms)':<17} | {'Rendimiento':<12} | {'Margen Error':<12}")
    print("-" * 80)
    
    mediana_ucs_nodos = np.median(resultados["UCS"]["nodos"])
    prom_ucs_tiempo = np.mean(resultados["UCS"]["tiempo"])
    costos_base = np.array(resultados["UCS"]["costo"])
    
    print(f"{'UCS (Baseline)':<16} | {int(mediana_ucs_nodos):<15} | {prom_ucs_tiempo:<17.2f} | {'1.00x':<12} | {'0.00%':<12}")
    
    for eq in etiquetas_astar:
        mediana_nodos = np.median(resultados[eq]["nodos"])
        prom_tiempo = np.mean(resultados[eq]["tiempo"])
        speedup = mediana_ucs_nodos / mediana_nodos
        
        costos_actuales = np.array(resultados[eq]["costo"])
        error_porcentual = np.mean(((costos_actuales - costos_base) / costos_base) * 100)
        
        print(f"{eq:<16} | {int(mediana_nodos):<15} | {prom_tiempo:<17.2f} | {f'{speedup:.2f}x':<12} | {f'{error_porcentual:.2f}%':<12}")
        
    print("-" * 80)
    input("\n[Presiona Enter para pasar a la Fase 4: Visualización Gráfica...]")
    generar_visualizaciones(resultados, etiquetas_astar)


def generar_visualizaciones(resultados, etiquetas_astar):
    print("\n" + "="*70)
    print("NIVEL 2 - FASE 4: VISUALIZACIÓN DE DISTRIBUCIONES DE RENDIMIENTO")
    print("="*70)
    
    etiquetas_plot = ["UCS"] + etiquetas_astar
    # PREPARACIÓN DE DATOS (FIX PARA ERROR DE TIPO DE DATO)
    data_list = []
    for alg in etiquetas_plot:
        for val in resultados[alg]["nodos"]:
            data_list.append({"Algoritmo": alg, "Nodos": val})
    
    df = pd.DataFrame(data_list) # Convertimos a DataFrame para Seaborn
    
    # FIGURA 1: TRADE-OFF
    costos_base = np.array(resultados["UCS"]["costo"])
    plt.figure(figsize=(10, 6))
    for alg in etiquetas_plot:
        err = np.mean(((np.array(resultados[alg]["costo"]) - costos_base) / costos_base) * 100)
        rend = np.median(resultados["UCS"]["nodos"]) / np.median(resultados[alg]["nodos"])
        plt.scatter(err, rend, label=alg, s=100)
    plt.xlabel("Margen de error (%)"); plt.ylabel("Rendimiento (aceleración)"); plt.title("Figura 1: Margen de error vs Rendimiento"); plt.legend(); plt.grid(True)
    plt.show()

    # FIGURA 2: COMPLEJIDAD ESPACIAL
    plt.figure(figsize=(10, 6))
    medias = [np.median(resultados[alg]["nodos"]) for alg in etiquetas_plot]
    plt.barh(etiquetas_plot, medias, color='skyblue')
    plt.xscale('log'); plt.title("Figura 2: Complejidad Espacial (escala logarítmica) "); plt.xlabel("Nodos promedios (Log)"); plt.show()
    
    

    # FIGURA 3: BOXPLOT (CORREGIDO PARA SEABORN v0.14.0+)
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df, 
        x="Algoritmo", 
        y="Nodos", 
        hue="Algoritmo",      # Asignamos el color a la misma variable del eje X
        palette="pastel",     # Definimos la paleta
        legend=False          # Ocultamos la leyenda redundante
    )
    plt.title("Figura 3: Distribución de Complejidad (Boxplot Monte Carlo)")
    plt.ylabel("Cantidad de Nodos")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()

    print("\n-> Gráficos generados. Análisis finalizado. GRACIAS")

if __name__ == "__main__":
    ejecutar_demostracion_unitaria()
