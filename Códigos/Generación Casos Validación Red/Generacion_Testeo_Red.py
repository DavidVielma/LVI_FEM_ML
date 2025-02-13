import numpy as np
from pyDOE import lhs
from collections import Counter
import itertools
import os
import random

def generar_combinaciones(capas_cf, capas_gf):
    combinaciones_posibles = list(itertools.permutations(['CF'] * capas_cf + ['GF'] * capas_gf))
    combinaciones_unicas = list(set(combinaciones_posibles))
    return combinaciones_unicas

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
    cantidad_total = 8  # Fijado a 8 para este caso específico (4 CF + 4 GF)

    # Generar todas las combinaciones posibles de materialidad
    capas_cf = 4
    capas_gf = 4
    combinaciones_posibles = generar_combinaciones(capas_cf, capas_gf)
    total_combinaciones_posibles = len(combinaciones_posibles)
    print(f"Número total de combinaciones posibles de materialidad: {total_combinaciones_posibles}")

    # Selección de materialidades por el usuario
    while True:
        num_materialidades = int(input(f"Ingrese la cantidad de materialidades a usar (máximo {total_combinaciones_posibles}): "))
        if num_materialidades <= total_combinaciones_posibles:
            break
        else:
            print(f"La cantidad ingresada es mayor al número de combinaciones posibles ({total_combinaciones_posibles}).")

    material_combinations = random.sample(combinaciones_posibles, num_materialidades)

    # Generación de ángulos usando LHS excluyendo los múltiplos de 10
    n_samples = int(input("Ingrese la cantidad total de samples para la variación de ángulos: "))
    n_angles = 3  # Ahora hay 3 ángulos variables
    angle_values = [i for i in range(1, 180) if i % 10 != 0]  # Excluir múltiplos de 10

    lhs_samples = lhs(n_angles, samples=n_samples)
    angle_combinations = np.array([[angle_values[int(val * (len(angle_values) - 1))] for val in sample] for sample in lhs_samples])

    # Crear combinaciones finales
    final_combinations = create_final_combinations(material_combinations, angle_combinations)

    # Guardar en archivo de texto
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

    # Mostrar la frecuencia de cada configuración de material
    material_counts = Counter([tuple([m[0] for m in comb[0]]) for comb in final_combinations])

    print("\nFrecuencia de cada configuración de material:")
    for material, count in material_counts.items():
        materials_str = f"[{','.join(material)}]"
        print(f'{materials_str}: {count} veces')

    print(f'Las combinaciones finales han sido guardadas en "{save_file_path}".')

if __name__ == "__main__":
    main()
