import numpy as np
from pyDOE import lhs
from collections import Counter
import random
import itertools
import os

def generar_combinaciones(capas_cf, capas_gf, total_combinaciones):
    combinaciones_posibles = list(itertools.permutations(['CF'] * capas_cf + ['GF'] * capas_gf))
    combinaciones_unicas = list(set(combinaciones_posibles))
    combinaciones_aleatorias = random.sample(combinaciones_unicas, total_combinaciones)
    combinaciones_formateadas = [",".join(combinacion) for combinacion in combinaciones_aleatorias]
    return combinaciones_formateadas

def parse_material_combinations(material_combinations_str):
    return [tuple(comb.split(',')) for comb in material_combinations_str]

def cargar_combinaciones_desde_archivo(file_path):
    with open(file_path, 'r') as file:
        combinaciones = [line.strip() for line in file.readlines()]
    return combinaciones

def aplicar_configuracion_angulos(angles):
    if len(angles) == 3:
        a1, a2, a3 = angles
        return [a1, a2, (180 - a1), a3, a3, (180 - a1), a2, a1]
    return [0] * 8

def create_final_combinations(material_combinations, angle_combinations):
    final_combinations = []

    for i in range(len(angle_combinations)):
        materials = list(material_combinations[i % len(material_combinations)])
        angles = aplicar_configuracion_angulos(angle_combinations[i])
        materials_with_angles = [(materials[j], angles[j]) for j in range(8)]
        final_combinations.append((materials_with_angles, angles))

    return final_combinations

def main():
    cantidad_total = 8  # Fijado a 8 para este caso específico

    file_path = input("Ingrese la ruta del archivo con las combinaciones de materiales (o presione Enter para generar aleatoriamente): ").strip()

    if file_path and os.path.exists(file_path):
        material_combinations_given = cargar_combinaciones_desde_archivo(file_path)
        print(f'Se han cargado {len(material_combinations_given)} combinaciones desde el archivo.')
    else:
        capas_cf = int(input("Ingrese la cantidad de capas de CF: "))
        capas_gf = int(input("Ingrese la cantidad de capas de GF: "))

        while capas_cf + capas_gf != cantidad_total:
            print(f"La suma de capas de CF y GF debe ser {cantidad_total}.")
            capas_cf = int(input("Ingrese la cantidad de capas de CF: "))
            capas_gf = int(input("Ingrese la cantidad de capas de GF: "))

        combinaciones_posibles = list(itertools.permutations(['CF'] * capas_cf + ['GF'] * capas_gf))
        combinaciones_unicas = list(set(combinaciones_posibles))
        total_combinaciones_posibles = len(combinaciones_unicas)
        print(f"Número total de combinaciones posibles: {total_combinaciones_posibles}")

        while True:
            total_combinaciones = int(input("Ingrese la cantidad de combinaciones de materiales deseadas: "))
            if total_combinaciones <= total_combinaciones_posibles:
                break
            else:
                print(f"La cantidad ingresada es mayor al número de combinaciones posibles ({total_combinaciones_posibles}).")

        material_combinations_given = generar_combinaciones(capas_cf, capas_gf, total_combinaciones)

    material_combinations = parse_material_combinations(material_combinations_given)

    n_samples = int(input("Ingrese la cantidad total de samples para la variación de ángulos: "))
    n_angles = 3  # Ahora hay 3 ángulos variables
    angle_values = np.arange(0, 171, 10)  # Ángulos de 0 a 180 grados, en pasos de 10

    lhs_samples = lhs(n_angles, samples=n_samples)
    angle_combinations = np.array([[angle_values[int(val * (len(angle_values)))] for val in sample] for sample in lhs_samples])

    final_combinations = create_final_combinations(material_combinations, angle_combinations)

    save_file_path = input("Ingrese la ruta y el nombre del archivo para guardar las combinaciones: ")

    with open(save_file_path, 'w') as file:
        for comb in final_combinations:
            material_layers = [m[0] for m in comb[0]]
            angles = comb[1]
            materials_str = f"[{','.join(material_layers)}]"
            angles_str = f"[{','.join(map(str, angles))}]"
            line = f'{materials_str} {angles_str}'
            print(line)
            file.write(line + '\n')

    material_counts = Counter([tuple([m[0] for m in comb[0]]) for comb in final_combinations])

    print("\nFrecuencia de cada configuración de material:")
    for material, count in material_counts.items():
        materials_str = f"[{','.join(material)}]"
        print(f'{materials_str}: {count} veces')

    print(f'Las combinaciones finales han sido guardadas en "{save_file_path}".')

if __name__ == "__main__":
    main()