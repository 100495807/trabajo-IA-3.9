from MFIS_Read_Functions import readFuzzySetsFile, readApplicationsFile, readRulesFile
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as skf


def generar_resultados():
    # leemos las reglas desde el archivo
    rules = readRulesFile()
    # leemos las aplicaciones desde el archivo
    applications = readApplicationsFile()
    # leemos los conjuntos difusos de las variables de entrada del archivo
    fuzzy_input_var_sets = readFuzzySetsFile("InputVarSets.txt")
    # leemos los conjuntos difusos de los riesgos desde el archivo
    fuzzy_risk = readFuzzySetsFile("Risks.txt")

    with open("results.txt", "w") as file:
        for app in applications:
            file.write(app.appId + " ")
            #calculamos los valores de riesgo
            valor_riesgo = logica_borrosa(rules, fuzzy_input_var_sets, app)
            #escribimos el valor de riesgo en el archivo
            file.write(f"lowR: {round(valor_riesgo['Risk=LowR'], 2)}, MediumR: {round(valor_riesgo['Risk=MediumR'], 2)}, HighR: {round(valor_riesgo['Risk=HighR'], 2)}\n")
            #calculo del centroide
            centroide = calc_centroide(valor_riesgo, fuzzy_risk)
            #escribimos el centroide en el archivo
            file.write(f"Centroid: {centroide}\n\n")


def logica_borrosa(rules, FuzzySetsDict, Application):
    risk = {"Risk=LowR": 0, "Risk=MediumR": 0, "Risk=HighR": 0}

    #para cada regla
    for rule in rules:
        #obtenemos los antecendentes y consecuentes de la regla
        lista_antecedente = rule.antecedent
        lista_consecuente = rule.consequent

        # inicializamos el grado de membresia
        grado_membresia_minimo = 1

        #para cada antecedente de la regla
        for ant in lista_antecedente:
            fuzzy = FuzzySetsDict[ant]

            # buscamos el valor de la variable del antecendente en los datos de la aplicacion
            for data in Application.data:
                if data[0] == fuzzy.var:
                    valor_antecedente = data[1]
                    break

            #calculamos el grado de membresia
            grado_membresia = skf.interp_membership(fuzzy.x, fuzzy.y, valor_antecedente)

            if grado_membresia < grado_membresia_minimo:
                grado_membresia_minimo = grado_membresia

        if risk[lista_consecuente] < grado_membresia_minimo:
            risk[lista_consecuente] = grado_membresia_minimo

    return risk


def calc_centroide(riesgo, fuzzy_risk):
    #primero se calcula el centroide de cada riesgo
    lowR = riesgo['Risk=LowR']
    mediumR = riesgo['Risk=MediumR']
    highR = riesgo['Risk=HighR']

    #almacenamos los valores de los conjuntos difusos de los riesgos
    lowR_x = fuzzy_risk['Risk=LowR'].x
    lowR_y = fuzzy_risk['Risk=LowR'].y
    mediumR_y = fuzzy_risk['Risk=MediumR'].y
    highR_y = fuzzy_risk['Risk=HighR'].y

    #lo segmentamos
    lowR_segment = np.minimum(lowR, lowR_y)
    mediumR_segment = np.minimum(mediumR, mediumR_y)
    highR_segment = np.minimum(highR, highR_y)

    #calculamos el centroide
    centroide = skf.defuzz(lowR_x, np.maximum(lowR_segment, np.maximum(mediumR_segment, highR_segment)), 'centroid')

    return centroide


def grafos():
    #leemos los conjuntos difusos de las variables de entrada
    fuzzy_input_var_sets = readFuzzySetsFile("InputVarSets.txt")
    #leemos los conjuntos difusos de los riesgos
    fuzzy_risk = readFuzzySetsFile("Risks.txt")

    #combianamos los dos diccionarios
    fuzzy_sets = {**fuzzy_input_var_sets, **fuzzy_risk}

    #inicializamos la lista de las variables que vamos a graficar
    vars_plot = []

    #para cada conjunto difuso
    for setId, fuzzySet in fuzzy_sets.items():
        #si el conjunto difuso es de una variable de entrada
        if fuzzySet.var not in vars_plot:
            if vars_plot:
                plt.xlabel(vars_plot[-1])
                plt.ylabel("Grado Membresia")
                plt.grid(True)
                plt.legend()
                plt.show()
            plt.figure()
            vars_plot.append(fuzzySet.var)
        plt.plot(fuzzySet.x, fuzzySet.y, label=fuzzySet.label)

    #ultima figura
    if vars_plot:
        plt.xlabel(vars_plot[-1])
        plt.ylabel("Grado Membresia")
        plt.grid(True)
        plt.legend()
        plt.show()


generar_resultados()
grafos()