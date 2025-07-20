# Ejemplos de Simulación — Universe

Este documento describe los principales scripts de simulación incluidos en `examples/`, sus resultados esperados y cómo interpretar los logs y visualizaciones.

---

## 1. Simulación acoplada de nucleones (visual, 2D)

**Script:** `simulate_coupled_nucleons_qt.py`

- Simula tres nucleones en 2D bajo fuerzas nucleares (Reid93) y de Coulomb.
- Permite elegir condiciones iniciales: colisión, lineal, triangular.
- Visualización en tiempo real con PyQtGraph.
- Logs de energía, fuerzas, aceleraciones y distancias.

**Ejemplo de ejecución:**
```bash
universe simulate --name simulate_coupled_nucleons_qt -- --init colision
```

**Interpretación:**
- Observa trayectorias, conservación de energía y respuesta a fuerzas acopladas.
- Los logs muestran la robustez física y la dinámica realista.

---

## 2. Tres nucleones en 2D (dinámica clásica)

**Script:** `simulate_three_nucleons_2d.py`

- Simulación de colisión y trayectorias de tres nucleones con potencial nuclear.
- Visualización de trayectorias y energía total.

---

## 3. Deuterón con potencial Reid93

**Script:** `simulate_deuteron_with_reid93.py`

- Simulación de un sistema de dos nucleones (deuterón) con potencial realista.
- Permite analizar la energía de enlace y la dinámica de pares nucleón-nucleón.

---

## 4. Partículas bajo fuerza de Coulomb

**Script:** `simulate_coulomb_particles_qt.py`

- Simulación visual de partículas cargadas bajo fuerza de Coulomb.
- Útil para validar la robustez del backend electromagnético.

---

## 5. Otros ejemplos y tests

- `plot_energy_levels.py`: Gráfica de niveles de energía cuánticos.
- `simulate_hadron_properties.py`: Propiedades de hadrones y validación de modelos de masa.

---

> Consulta los logs y visualizaciones de cada script para validar la física y la robustez numérica del framework.
