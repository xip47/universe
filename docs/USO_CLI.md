# Guía de uso del CLI — Universe

El CLI de Universe permite ejecutar simulaciones físicas, tests automáticos y consultas científicas desde la terminal de forma profesional y flexible.

---

## Comandos principales

- `universe list` — Lista todas las simulaciones disponibles en `examples/`.
- `universe simulate --name <script> [-- <args>]` — Ejecuta una simulación visual o de test.
- `universe test` — Ejecuta todos los tests automáticos del framework.
- `universe query mass --particle <nombre>` — Consulta la masa de una partícula del Modelo Estándar.
- `universe query mass --list` — Lista todas las partículas y sus masas.

---

## Ejemplos de uso

### Listar simulaciones
```bash
universe list
```

### Ejecutar una simulación visual acoplada
```bash
universe simulate --name simulate_coupled_nucleons_qt -- --init colision
```

Opciones para `--init`:
- `colision`   : Colisión asimétrica de tres nucleones
- `lineal`     : Tres nucleones alineados
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
universe query mass --particle neutron
universe query mass --list
```

---

## Argumentos y sintaxis avanzada

- Los argumentos después de `--` se pasan directamente al script de simulación.
- Puedes combinar opciones del CLI con argumentos personalizados para cada script.

---

> Consulta el README y la documentación de cada script en `examples/` para más detalles y opciones avanzadas.
