import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as skf
from MFIS_Read_Functions import readFuzzySetsFile, readApplicationsFile, readRulesFile



# Función para guardar contenido en un archivo
def guardar_contenido(archivo, contenido):
    """Esta función toma un nombre de archivo y contenido como
    entrada, y escribe el contenido en el archivo especificado."""

    with open(archivo, 'w') as archivo:
        archivo.write(contenido)


# Función principal para calcular riesgos
def calcular_riesgos():
    """Esta función es la función principal que lee las reglas, las aplicaciones y
    los conjuntos difusos de los archivos correspondientes. Calcula el riesgo para
    cada aplicación utilizando la lógica difusa y guarda los resultados en un archivo.
    También llama a la función generar_graficas() para generar gráficos para los conjuntos difusos."""

    # Leemos las reglas desde el archivo correspondiente
    reglas = readRulesFile()

    # Leemos las aplicaciones desde el archivo correspondiente
    aplicaciones = readApplicationsFile()

    # Leemos los conjuntos difusos de las variables de entrada desde el archivo correspondiente
    conjuntos_difusos_entrada = readFuzzySetsFile("InputVarSets.txt")

    # Leemos los conjuntos difusos de los riesgos desde el archivo correspondiente
    conjuntos_difusos_riesgo = readFuzzySetsFile("Risks.txt")

    # Inicializamos una lista para almacenar los resultados
    lista_resultados = []

    # Iteramos sobre cada aplicación
    for app in aplicaciones:
        # Calculamos el riesgo para la aplicación actual utilizando la lógica difusa
        valor_riesgo = calcular_logica_difusa(reglas, conjuntos_difusos_entrada, app)

        # Añadimos el riesgo calculado a la lista de resultados
        lista_resultados.append(
            f"{app.appId} lowR: {round(valor_riesgo['Risk=LowR'], 2)}, "
            f"MediumR: {round(valor_riesgo['Risk=MediumR'], 2)}, "
            f"HighR: {round(valor_riesgo['Risk=HighR'], 2)}")

        # Calculamos el centroide del riesgo
        centroide = calcular_centroide(valor_riesgo, conjuntos_difusos_riesgo)

        # Añadimos el centroide a la lista de resultados
        lista_resultados.append(f"Centroid: {centroide}\n\n")

    # Guardamos los resultados en un archivo
    guardar_contenido("results.txt", "\n".join(lista_resultados))


# Función para calcular la lógica difusa
def calcular_logica_difusa(reglas, FuzzySetsDict, Application):
    """Esta función toma un conjunto de reglas, un diccionario de conjuntos difusos
    y una aplicación como entrada. Calcula y devuelve el riesgo asociado con la aplicación
    utilizando lógica difusa."""

    # Inicializamos un diccionario para almacenar los riesgos
    riesgo = {"Risk=LowR": 0, "Risk=MediumR": 0, "Risk=HighR": 0}

    # Iteramos sobre cada regla
    for regla in reglas:
        # Obtenemos la lista de antecedentes y consecuentes de la regla
        lista_antecedente = regla.antecedent
        lista_consecuente = regla.consequent

        # Inicializamos el grado de membresía mínimo
        grado_membresia_minimo = 1

        # Iteramos sobre cada antecedente en la lista de antecedentes
        for antecendente in lista_antecedente:
            # Obtenemos el conjunto difuso correspondiente al antecedente
            conjunto_difuso = FuzzySetsDict[antecendente]

            # Iteramos sobre los datos de la aplicación
            for dato in Application.data:
                # Si el nombre de la variable coincide con el del conjunto difuso
                if dato[0] == conjunto_difuso.var:
                    # Obtenemos el valor del antecedente
                    valor_antecedente = dato[1]
                    break

            # Calculamos el grado de membresía del valor del antecedente en el conjunto difuso
            grado_membresia = skf.interp_membership(conjunto_difuso.x, conjunto_difuso.y, valor_antecedente)

            # Si el grado de membresía es menor que el grado de membresía mínimo actual
            if grado_membresia < grado_membresia_minimo:
                # Actualizamos el grado de membresía mínimo
                grado_membresia_minimo = grado_membresia

        # Si el grado de membresía mínimo es mayor que el riesgo actual para el consecuente
        if riesgo[lista_consecuente] < grado_membresia_minimo:
            # Actualizamos el riesgo para el consecuente
            riesgo[lista_consecuente] = grado_membresia_minimo

    # Devolvemos el diccionario de riesgos
    return riesgo


# Función para calcular el centroide
def calcular_centroide(riesgo, conjuntos_difusos_riesgo):
    """Esta función toma un riesgo y un conjunto de conjuntos difusos de
    riesgo como entrada. Calcula y devuelve el centroide del riesgo"""

    # Extraemos los valores de riesgo bajo, medio y alto del diccionario de riesgo
    riesgo_bajo = riesgo['Risk=LowR']
    riesgo_medio = riesgo['Risk=MediumR']
    riesgo_alto = riesgo['Risk=HighR']

    # Extraemos los conjuntos difusos de riesgo bajo, medio y alto
    x_riesgo_bajo = conjuntos_difusos_riesgo['Risk=LowR'].x
    y_riesgo_bajo = conjuntos_difusos_riesgo['Risk=LowR'].y
    y_riesgo_medio = conjuntos_difusos_riesgo['Risk=MediumR'].y
    y_riesgo_alto = conjuntos_difusos_riesgo['Risk=HighR'].y

    # Calculamos los segmentos de riesgo bajo, medio y alto
    segmento_riesgo_bajo = np.minimum(riesgo_bajo, y_riesgo_bajo)
    segmento_riesgo_medio = np.minimum(riesgo_medio, y_riesgo_medio)
    segmento_riesgo_alto = np.minimum(riesgo_alto, y_riesgo_alto)

    # Calculamos el centroide utilizando la función defuzz de la biblioteca skfuzzy
    centroide = skf.defuzz(x_riesgo_bajo,
                           np.maximum(segmento_riesgo_bajo, np.maximum(segmento_riesgo_medio, segmento_riesgo_alto)),
                           'centroid')

    # Devolvemos el valor del centroide
    return centroide


# Función para generar gráficas
def generar_graficas():
    """Esta función lee los conjuntos difusos de las variables de entrada y los riesgos
     de los archivos correspondientes. Genera y muestra gráficos para cada conjunto difuso."""

    # Leemos los conjuntos difusos de las variables de entrada desde el archivo "InputVarSets.txt"
    conjuntos_difusos_entrada = readFuzzySetsFile("InputVarSets.txt")

    # Leemos los conjuntos difusos de los riesgos desde el archivo "Risks.txt"
    conjuntos_difusos_riesgo = readFuzzySetsFile("Risks.txt")

    # Combinamos los dos conjuntos difusos en un solo diccionario
    conjuntos_difusos = {**conjuntos_difusos_entrada, **conjuntos_difusos_riesgo}

    # Inicializamos una lista para almacenar las variables que vamos a graficar
    variables_a_graficar = []

    # Iteramos sobre cada conjunto difuso en el diccionario
    for setId, conjunto_difuso in conjuntos_difusos.items():
        # Si la variable del conjunto difuso actual no está en la lista de variables a graficar
        if conjunto_difuso.var not in variables_a_graficar:
            # Si ya hay variables en la lista a graficar, mostramos la gráfica actual
            if variables_a_graficar:
                plt.xlabel(variables_a_graficar[-1])
                plt.ylabel("Grado Membresia")
                plt.grid(True)
                plt.legend()
                plt.show()
            # Creamos una nueva figura para la próxima gráfica
            plt.figure()
            # Añadimos la variable del conjunto difuso actual a la lista de variables a graficar
            variables_a_graficar.append(conjunto_difuso.var)
        # Graficamos el conjunto difuso actual
        plt.plot(conjunto_difuso.x, conjunto_difuso.y, label=conjunto_difuso.label)

    # Si hay variables en la lista a graficar, mostramos la última gráfica
    if variables_a_graficar:
        plt.xlabel(variables_a_graficar[-1])
        plt.ylabel("Grado Membresia")
        plt.grid(True)
        plt.legend()
        plt.show()


# Llamada a las funciones principales
calcular_riesgos()
generar_graficas()
