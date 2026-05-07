# trabajo-IA-3.9

Proyecto de Inteligencia Artificial centrado en un sistema de inferencia difusa para estimar riesgo en solicitudes. El repositorio incluye definicion de conjuntos difusos, base de reglas, aplicaciones de prueba, resultados, graficas y documentacion de entrega.

## Objetivo

Implementar un sistema MFIS (Mamdani Fuzzy Inference System) que reciba variables de entrada como edad, ingresos, patrimonio, cantidad solicitada, tipo de trabajo e historial, y devuelva una estimacion de riesgo.

## Tecnologias

- Python
- NumPy
- scikit-fuzzy
- matplotlib
- Logica difusa
- Ficheros de entrada en texto plano

## Contenido

| Archivo / Carpeta | Descripcion |
| --- | --- |
| `MFIS_Classes.py` | Clases para conjuntos difusos, reglas y aplicaciones. |
| `MFIS_Read_Functions.py` | Lectura de conjuntos, reglas y aplicaciones. |
| `resultados.py` | Evaluacion y generacion de resultados/graficas. |
| `InputVarSets.txt` | Variables y conjuntos difusos de entrada. |
| `Rules.txt` | Base de reglas IF-THEN. |
| `Risks.txt` | Conjuntos difusos de salida. |
| `Applications.txt` | Casos de entrada a evaluar. |
| `results.txt` | Resultados producidos. |
| `foto_graficas/` | Graficas de variables difusas. |
| `Practica IA 2024 G-06/` | Entregables finales: informe, analisis, codigo y video. |

## Funcionamiento

1. Se cargan conjuntos difusos trapezoidales.
2. Se leen reglas de decision desde `Rules.txt`.
3. Se procesan aplicaciones de entrada.
4. Se calcula la activacion de reglas segun grados de pertenencia.
5. Se obtiene una salida de riesgo y se generan resultados/graficas.

## Como Ejecutarlo

```bash
pip install numpy scikit-fuzzy matplotlib
python resultados.py
```

## Aprendizajes

- Modelar incertidumbre con conjuntos difusos.
- Separar reglas, datos y motor de inferencia.
- Leer conocimiento experto desde ficheros externos.
- Visualizar funciones de pertenencia para validar el sistema.
- Producir resultados trazables para casos de prueba.

## Estado

Proyecto academico finalizado. Se conserva como practica de inferencia difusa aplicada a evaluacion de riesgo.