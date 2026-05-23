# Explorador de Mapas IA: Benchmarking de Algoritmos de Búsqueda

Este repositorio contiene el framework experimental desarrollado para evaluar la eficiencia y el rendimiento de algoritmos de búsqueda en entornos procedurales. El sistema compara la **Búsqueda de Costo Uniforme (UCS)** contra una implementación optimizada de **A***, utilizando una heurística original basada en la esperanza matemática del terreno.

## 👥 Equipo de Investigación
- Cáceres Flores, Abimael Fernando
- Luque Canaza, Glenn Rozier
- Perez Aguinaga, Jheyder Whitman
- Romero Lovera, Junio Alberto

**Profesor:** Yvan Jesus Tupac Valdivia
**Curso:** Fundamentos de la Inteligencia Artificial | UNI 2026

---

## 🚀 Descripción del Proyecto
El programa implementa un motor de generación procedural de mapas ($100 \times 100$) con tres tipos de terreno (Llanura, Bosque, Obstáculo) y evalúa el comportamiento de agentes autónomos bajo dos fases:
1. **Fase 1 (Caso Base):** Visualización técnica del recorrido óptimo y métricas de desempeño individual.
2. **Fase 2 (Simulación de Montecarlo):** Benchmarking riguroso sobre 500 iteraciones para análisis de sensibilidad y distribución estadística de la complejidad espacial/temporal.

---

## 🛠️ Requisitos e Instalación

El proyecto utiliza librerías estándar para análisis de datos y visualización científica. Asegúrate de tener instalado Python 3.8 o superior.

### Instalación de dependencias
Ejecuta el siguiente comando en tu terminal para instalar los paquetes necesarios:

```bash
pip install numpy matplotlib seaborn pandas