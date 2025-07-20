# Universe

**Simulación física completa del universo**, desde partículas fundamentales hasta la vida emergente, con máxima fidelidad científica, modularidad y escalabilidad.
Incluye soporte para computación en GPU (CuPy), simulaciones cuánticas, nucleares, electromagnéticas y biológicas futuras.

---

## Características principales

- **Física realista:** Modelado de partículas, fuerzas nucleares (Reid93, Yukawa), electromagnetismo (Coulomb, Maxwell), grados de libertad extendidos (spin, isospin, carga, estado cuántico).
- **Unidades naturales:** Todas las simulaciones usan MeV, fm, MeV/fm, MeV/c², e, c=1.
- **Backend GPU:** Computación acelerada con CuPy (CUDA) o NumPy.
- **Visualización avanzada:** Simulaciones interactivas en 2D/3D con PyQtGraph y PyQt5.
- **CLI profesional:** Ejecuta simulaciones, tests y consultas físicas desde la terminal.
- **Arquitectura escalable:** Preparado para acoplar nuevos dominios físicos y sistemas complejos.

---

## Requisitos

- Python 3.12
- CUDA Toolkit 12.x (para GPU)
- NVIDIA GPU compatible
- CuPy (`cupy-cuda12x`)
- PyQt5 (visualización avanzada)
- Matplotlib (opcional, para gráficos estáticos)

---

## Instalación

```bash
pip install .
```

---

## Uso del CLI

### Listar simulaciones disponibles

```bash
universe list
```

### Ejecutar una simulación visual (ejemplo: nucleones acoplados)

```bash
universe simulate --name simulate_coupled_nucleons_qt -- --init colision
```

Opciones para `--init`:
- `colision`   : Colisión asimétrica de tres nucleones (dinámica rica)
- `lineal`     : Tres nucleones alineados con velocidades alternas
- `triangular` : Configuración triangular (por defecto)

### Ejecutar otros ejemplos

```bash
universe simulate --name simulate_three_nucleons_2d
universe simulate --name simulate_deuteron_with_reid93
universe simulate --name simulate_coulomb_particles_qt
```

### Ejecutar todos los tests

```bash
universe test
```

### Consultar masas del Modelo Estándar

```bash
universe query mass --particle proton
universe query mass --list
```

---

## Ejemplo de simulación visual acoplada

```bash
universe simulate --name simulate_coupled_nucleons_qt -- --init colision
```
- Visualiza trayectorias, energía, fuerzas y aceleraciones de nucleones bajo fuerzas nucleares y de Coulomb.
- Logs detallados de energía, distancias, fuerzas y aceleraciones en tiempo real.
- Compatible con GPU y CPU.

---

## Estructura del proyecto

- `universe/`         : Núcleo del framework, física, partículas, sistemas, solvers, numerics, etc.
- `examples/`         : Scripts de simulación visual y de test listos para ejecutar.
- `tests/`            : Tests automáticos de validación física y numérica.
- `docs/`             : Documentación y criterios de desarrollo.

---

## Estándares y buenas prácticas

- **Tipado estricto Python 3** y documentación NumPy en español.
- **Imports individuales** y arquitectura modular.
- **Unidades naturales** en todo el código (MeV, fm, c=1).
- **Cobertura de tests** y validación física en cada módulo.
- **Preparado para extensiones**: nuevos dominios físicos, IA, vida emergente.

---

## Créditos y contacto

Desarrollado por [Tu Nombre/Equipo].
Basado en literatura científica y criterios de máxima fidelidad física.
