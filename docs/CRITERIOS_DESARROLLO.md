# Criterios de Desarrollo y Estándares de Programación para Universe

## 1. Principios Fundamentales

1. **Fidelidad absoluta a la realidad física**
   - Cada elemento, ley, regla, interacción y partícula debe modelar la naturaleza de la forma más precisa posible según el conocimiento científico actual (PDG, literatura revisada, etc.).
2. **Escalabilidad**
   - El software debe ser modular y preparado para crecer desde partículas elementales hasta la vida emergente y sistemas complejos, permitiendo la integración de nuevas leyes, interacciones y dominios físicos.
3. **Optimización y performance**
   - Todos los cálculos deben estar optimizados, priorizando el uso de CUDA y computación paralela para aprovechar al máximo el hardware disponible.
4. **Testeabilidad**
   - Cada funcionalidad, ley física, interacción y módulo debe tener tests automáticos y simulaciones que demuestren su correcto funcionamiento y fidelidad.
5. **Soluciones definitivas**
   - Evitar parches o soluciones temporales; cada implementación debe ser robusta, mantenible y pensada para el largo plazo.
6. **Inteligencia artificial**
   - Solo se introducirá IA una vez que se haya logrado simular la vida emergente de manera fidedigna.

## 2. Reglas de Programación y Estilo

- **Tipificación estricta**: Todo el código debe estar completamente tipado usando las anotaciones de Python 3.
- **Importaciones individuales**: Cada módulo debe importar solo lo necesario, usando una línea por importación.
- **Identación legible**: Usar 4 espacios por nivel de identación, nunca tabs.
- **Documentación tipo NumPy**: Todas las funciones, clases y métodos deben estar documentados con docstrings en formato NumPy, con cabeceras en inglés y descripciones en español.
- **Nombres descriptivos**: Variables, funciones y clases deben tener nombres claros y representativos de su propósito físico o computacional.
- **Código modular**: Separar la lógica en módulos y submódulos según el dominio físico o funcional.
- **Evitar duplicidad**: Reutilizar funciones y utilidades comunes, evitando la repetición de código.
- **Control de errores explícito**: Validar argumentos y estados, lanzando excepciones claras y documentadas.
- **Tests exhaustivos**: Cada módulo debe tener su propio archivo de tests, cubriendo casos normales, límites y errores esperados.
- **Cobertura de código**: Mantener una cobertura de tests superior al 90% en los módulos críticos.
- **Estilo consistente**: Seguir PEP8 y Black para el formateo automático del código.
- **Revisiones de código**: Todo cambio debe ser revisado y aprobado antes de integrarse a la rama principal.

## 3. Ejemplo de Docstring (formato NumPy, cabecera en inglés)

```python
from typing import Any

def kinetic_energy(mass: float, velocity: float) -> float:
    """
    Compute the classical kinetic energy of a particle.

    Calcula la energía cinética clásica de una partícula.

    Parameters
    ----------
    mass : float
        Masa de la partícula en kilogramos (kg).
    velocity : float
        Velocidad de la partícula en metros por segundo (m/s).

    Returns
    -------
    float
        Energía cinética en julios (J).
    """
    return 0.5 * mass * velocity ** 2
```

## 4. Validación y Simulación

- Cada nueva ley, interacción o funcionalidad debe ir acompañada de:
  - Un test unitario que valide su comportamiento físico y computacional.
  - Una simulación de ejemplo que demuestre su uso y resultados esperados.

## 5. Actualización de Criterios

- Este documento debe ser actualizado cada vez que se introduzca un nuevo estándar, regla o principio relevante para el desarrollo del framework Universe.
