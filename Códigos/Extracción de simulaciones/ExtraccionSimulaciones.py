import os
import csv
from ansys.dpf import core as dpf
import tkinter as tk
from tkinter import filedialog

# Función para seleccionar carpeta usando un cuadro de diálogo
def select_folder(title):
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    folder_path = filedialog.askdirectory(title=title)
    return folder_path

# Solicitar al usuario la carpeta madre y el archivo CSV de salida
base_directory = select_folder("Selecciona la carpeta madre que contiene las subcarpetas de simulación")
output_csv_path = filedialog.asksaveasfilename(defaultextension=".csv", title="Guardar archivo CSV de salida")

# Función para procesar nombres de subsubcarpetas y extraer información
def parse_folder_name(folder_name):
    parts = folder_name.split('_')
    energy = parts[0]
    materials = parts[1:9]  # Las primeras 8 capas son los materiales
    angles = parts[9:]  # Los ángulos son el resto
    return energy, materials, angles

# Función para contar los elementos dañados en cada archivo d3plot
def process_simulation(folder_path, itime=101):
    # Cargar el modelo d3plot
    d3plot_file = os.path.join(folder_path, 'd3plot')

    # Verificar si el archivo d3plot existe
    if not os.path.exists(d3plot_file):
        print(f"Error: Archivo d3plot no encontrado en: {d3plot_file}")
        return None

    print(f"Procesando archivo: {d3plot_file}")

    try:
        model = dpf.Model(d3plot_file)
    except Exception as e:
        print(f"Error al cargar el archivo d3plot: {e}")
        return None

    # Obtener las variables históricas y el desplazamiento en el tiempo especificado
    try:
        hist = getattr(model.results, "history_variablesihv__[1__5]").on_time_scoping([itime]).eval()
    except Exception as e:
        print(f"Error al obtener variables históricas: {e}")
        return None

    hist1 = hist[0].data[:120000]
    hist2 = hist[1].data[:120000]
    hist3 = hist[2].data[:120000]
    hist4 = hist[3].data[:120000]
    hist5 = hist[4].data[:120000]

    damages = [
        120000 - sum(hist1),
        120000 - sum(hist2),
        120000 - sum(hist3),
        120000 - sum(hist4),
        120000 - sum(hist5)
    ]

    return damages

# Inicializar la lista de datos para el CSV
output_data = []
simulation_number = 1

# Recorrer cada subcarpeta
for subfolder in os.listdir(base_directory):
    subfolder_path = os.path.join(base_directory, subfolder)

    if os.path.isdir(subfolder_path):  # Verificar que sea una carpeta
        # Recorrer cada subsubcarpeta dentro de la subcarpeta
        for subsubfolder in os.listdir(subfolder_path):
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)

            if os.path.isdir(subsubfolder_path):  # Verificar que sea una carpeta
                # Extraer información del nombre de la subsubcarpeta
                energy, materials, angles = parse_folder_name(subsubfolder)

                # Procesar la simulación
                damages = process_simulation(subsubfolder_path)

                if damages is not None:
                    # Almacenar la información en el CSV
                    row = [simulation_number] + materials + angles + [energy] + damages
                    output_data.append(row)
                    simulation_number += 1

# Escribir los resultados en un archivo CSV
with open(output_csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Escribir el encabezado
    headers = ['Simulacion'] + [f'Material_{i+1}' for i in range(8)] + [f'Angulo_{i+1}' for i in range(8)] + ['Energia', 'Damage_1', 'Damage_2', 'Damage_3', 'Damage_4', 'Damage_5']
    csv_writer.writerow(headers)

    # Escribir los datos
    csv_writer.writerows(output_data)

print(f"Datos almacenados exitosamente en {output_csv_path}")
