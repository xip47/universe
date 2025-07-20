# Guía de Unidades y Convenciones Físicas — Universe

Universe utiliza **unidades naturales** para máxima fidelidad y robustez en simulaciones físicas.

---

## Tabla de unidades principales

| Magnitud         | Unidad usada | Símbolo | Equivalencia SI           |
|------------------|-------------|---------|---------------------------|
| Energía          | MeV         | MeV     | 1 MeV = 1.60218e-13 J     |
| Longitud         | fm          | fm      | 1 fm = 1e-15 m            |
| Masa             | MeV/c²      | MeV/c²  | 1 MeV/c² = 1.78266e-30 kg |
| Carga eléctrica  | e           | e       | 1 e = 1.60218e-19 C       |
| Velocidad        | c=1         | fm/c    | c = 299792458 m/s         |
| Fuerza           | MeV/fm      | MeV/fm  |                           |

---

## Ejemplos de conversión

- Energía: `E [MeV] = E [J] / 1.60218e-13`
- Longitud: `L [fm] = L [m] * 1e15`
- Masa: `m [MeV/c²] = m [kg] / 1.78266e-30`

---

## Buenas prácticas

- **Nunca mezcles unidades SI y naturales** en una misma simulación.
- Usa siempre las constantes y conversiones centralizadas en `physics/constants.py`.
- Documenta explícitamente las unidades de entrada y salida en cada función/módulo.
- Prefiere `xp` (NumPy/CuPy) para arrays y operaciones vectorizadas.

---

## Advertencias

- La fidelidad física depende de la consistencia de unidades en todo el flujo.
- Si integras módulos externos, revisa y adapta las unidades antes de acoplarlos.

---

> Universe está preparado para simulaciones de alta fidelidad gracias a la adopción rigurosa de unidades naturales.
