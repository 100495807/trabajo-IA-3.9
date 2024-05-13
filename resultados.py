import matplotlib.pyplot as plt
from MFIS_Read_Functions import readRulesFile, readApplicationsFile, readFuzzySetsFile
import numpy as np
import skfuzzy as skf

def create_results():
    # Carga los datos de las reglas, aplicaciones y conjuntos difusos
    rules = readRulesFile()
    applications = readApplicationsFile()
    input_fuzzy_sets = readFuzzySetsFile("InputVarSets.txt")
    risk_fuzzy_sets = readFuzzySetsFile("Risks.txt")
    risk_outcomes = []

    # Itera sobre cada aplicación
    for application in applications:
        # Calcula los valores de riesgo
        risk_outcome = calculate_fuzzy_logic(rules, input_fuzzy_sets, application)
        # Guarda los resultados
        risk_outcomes.append(f"{application.appId} LowR: {risk_outcome['Risk=LowR']}, "
                             f"MediumR: {risk_outcome['Risk=MediumR']}, "
                             f"HighR: {risk_outcome['Risk=HighR']}")
        # Calcula el centroide
        centroid_outcome = compute_centroid(risk_outcome, risk_fuzzy_sets)
        # Guarda el centroide
        risk_outcomes.append(f"Centroid: {centroid_outcome}\n")

    # Guarda los resultados en un archivo
    save_to_file("results.txt", "\n".join(risk_outcomes))

def calculate_fuzzy_logic(rules, fuzzy_sets, application):
    # Inicializa los grados de membresía de los riesgos
    risk_membership = {"Risk=LowR": 0, "Risk=MediumR": 0, "Risk=HighR": 0}

    # Itera sobre cada regla
    for rule in rules:
        min_membership = 1

        # Calcula el grado de membresía mínimo para cada antecedente de la regla
        for antecedent in rule.antecedent:
            fuzzy_set = fuzzy_sets[antecedent]

            # Encuentra el valor del antecedente en los datos de la aplicación
            antecedent_value = next(data[1] for data in application.data if data[0] == fuzzy_set.var)

            # Calcula el grado de membresía y actualiza el mínimo
            membership_value = skf.interp_membership(fuzzy_set.x, fuzzy_set.y, antecedent_value)
            min_membership = min(min_membership, membership_value)

        # Actualiza los grados de membresía de riesgo si es necesario
        if risk_membership[rule.consequent] < min_membership:
            risk_membership[rule.consequent] = min_membership

    return risk_membership

def compute_centroid(risk_membership, risk_fuzzy_sets):
    # Inicializa las listas para almacenar los segmentos de cada riesgo
    risk_segments = []

    # Itera sobre los riesgos y sus grados de membresía
    for risk, membership in risk_membership.items():
        # Obtiene los puntos de las funciones de membresía del riesgo
        x_points = risk_fuzzy_sets[risk].x
        y_points = risk_fuzzy_sets[risk].y
        # Encuentra el segmento entre la membresía calculada y la función de membresía del riesgo
        segment = np.minimum(membership, y_points)
        # Añade el segmento a la lista
        risk_segments.append(segment)

    # Calcula el centroide utilizando la regla del centroide
    centroid = skf.defuzz(x_points, np.maximum.reduce(risk_segments), 'centroid')

    return centroid

def save_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

def create_graphics():
    # Lee los conjuntos difusos
    input_fuzzy_sets = readFuzzySetsFile("InputVarSets.txt")
    risk_fuzzy_sets = readFuzzySetsFile("Risks.txt")
    combined_sets = {**input_fuzzy_sets, **risk_fuzzy_sets}
    plotted_vars = []

    # Itera sobre cada conjunto difuso y genera una gráfica
    for set_id, fuzzy_set in combined_sets.items():
        if fuzzy_set.var not in plotted_vars:
            if plotted_vars:
                plt.legend()
                plt.show()
            plt.figure()
            plt.xlabel(fuzzy_set.var)
            plt.ylabel("Grado Membresia")
            plt.grid(True)
            plotted_vars.append(fuzzy_set.var)
        plt.plot(fuzzy_set.x, fuzzy_set.y, label=fuzzy_set.label)

    # Muestra la última figura
    if plotted_vars:
        plt.legend()
        plt.show()

# Genera los resultados y las gráficas
create_results()
'''create_graphics()
'''