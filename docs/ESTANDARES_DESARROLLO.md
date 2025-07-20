# Estándares de Desarrollo y Contribución — Universe

## Principios fundamentales

- **Fidelidad física absoluta:** Cada módulo debe modelar la naturaleza con máxima precisión científica.
- **Escalabilidad y modularidad:** El código debe ser fácil de extender y mantener.
- **Performance y GPU:** Priorizar el uso de CuPy y computación paralela.
- **Testeabilidad:** Todo debe tener tests automáticos y simulaciones de validación.
- **Soluciones definitivas:** Evitar parches temporales; priorizar robustez y mantenibilidad.

---

## Reglas de codificación

- **Tipado estricto:** Usa anotaciones de tipos Python 3 en todo el código.
- **Documentación NumPy en español:** Docstrings con headers en inglés y descripciones en español.
- **Imports individuales:** Una línea por importación, solo lo necesario.
- **Indentación legible:** 4 espacios por nivel, nunca tabs.
- **Nombres descriptivos:** Variables, funciones y clases deben ser claros y representativos.

---

## Testing y validación

- **Cobertura de tests:** Cada módulo debe tener tests automáticos en `tests/`.
- **Validación física:** Simulaciones y logs deben demostrar conservación de energía y robustez física.
- **Tests de integración:** Para sistemas acoplados y flujos completos.

---

## Buenas prácticas

- Usa el backend `xp` (NumPy/CuPy) para arrays y operaciones vectorizadas.
- Centraliza constantes y conversiones en `physics/constants.py`.
- Documenta explícitamente las unidades en cada función/módulo.
- Mantén la bitácora de checkpoints y snapshots de avance.

---

## Cómo contribuir

1. **Sigue los estándares anteriores** en cada PR o commit.
2. **Agrega o actualiza tests** para cada nueva funcionalidad.
3. **Actualiza la documentación** si cambias la API o la arquitectura.
4. **Incluye ejemplos de uso** en `examples/` si agregas nuevas capacidades.
5. **Mantén la trazabilidad**: actualiza la bitácora en `docs/checkpoints/`.

---

> Universe es un framework abierto y robusto: tu contribución es bienvenida si sigues estos estándares.
