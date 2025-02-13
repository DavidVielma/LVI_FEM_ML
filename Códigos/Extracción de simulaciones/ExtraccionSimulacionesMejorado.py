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

# Función para solicitar un prefijo al usuario
def get_prefix():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    return filedialog.asksaveasfilename(title="Ingresa el prefijo para los archivos", defaultextension="")

# Solicitar al usuario la carpeta madre y el prefijo para los archivos CSV
base_directory = select_folder("Selecciona la carpeta madre que contiene las subcarpetas de simulación")
prefix = get_prefix()

# Función para procesar nombres de subsubcarpetas y extraer información
def parse_folder_name(folder_name):
    parts = folder_name.split('_')
    energy = parts[0]
    materials = parts[1:9]  # Las primeras 8 capas son los materiales
    angles = parts[9:]  # Los ángulos son el resto
    return energy, materials, angles

# Función para procesar simulación y obtener datos
def process_simulation(folder_path, itime=2):
    d3plot_file = os.path.join(folder_path, 'd3plot')

    if not os.path.exists(d3plot_file):
        print(f"Error: Archivo d3plot no encontrado en: {d3plot_file}")
        return None

    print(f"Procesando archivo: {d3plot_file}")

    try:
        model = dpf.Model(d3plot_file)
    except Exception as e:
        print(f"Error al cargar el archivo d3plot: {e}")
        return None

    try:
        hist = getattr(model.results, "history_variablesihv__[1__5]").on_time_scoping([itime]).eval()
    except Exception as e:
        print(f"Error al obtener variables históricas: {e}")
        return None

    hist_data = [h.data[:120000] for h in hist]

    damages = [
        120000 - sum(hist_data[0]),
        120000 - sum(hist_data[1]),
        120000 - sum(hist_data[2]),
        120000 - sum(hist_data[3]),
        120000 - sum(hist_data[4])
    ]

    # Convertir los datos de hist a 0 o 1 para reducir el tamaño del archivo
    hist_data = [[int(value) for value in h] for h in hist_data]

    return hist_data, damages

# Inicializar datos para los archivos CSV
res_danos_config = []
hist_csvs = [[] for _ in range(5)]  # 5 listas para hist1 a hist5
simulation_number = 1

# Recorrer subcarpetas
for subfolder in os.listdir(base_directory):
    subfolder_path = os.path.join(base_directory, subfolder)

    if os.path.isdir(subfolder_path):
        for subsubfolder in os.listdir(subfolder_path):
            subsubfolder_path = os.path.join(subfolder_path, subsubfolder)

            if os.path.isdir(subsubfolder_path):
                energy, materials, angles = parse_folder_name(subsubfolder)
                result = process_simulation(subsubfolder_path)

                if result is not None:
                    hist_data, damages = result

                    # Agregar a ResDanosConfig
                    res_danos_config.append([simulation_number] + materials + angles + [energy] + damages)

                    # Agregar a los archivos individuales
                    for i, hist in enumerate(hist_data):
                        hist_csvs[i].append([energy] + materials + angles + hist)

                    simulation_number += 1

# Escribir el archivo ResDanosConfig.csv
with open(f"{prefix}_ResDanosConfig.csv", mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    headers = ['Simulacion'] + [f'Material_{i+1}' for i in range(8)] + [f'Angulo_{i+1}' for i in range(8)] + ['Energia', 'Damage_1', 'Damage_2', 'Damage_3', 'Damage_4', 'Damage_5']
    csv_writer.writerow(headers)
    csv_writer.writerows(res_danos_config)

# Escribir los archivos hist CSV
for i, hist_csv in enumerate(hist_csvs):
    with open(f"{prefix}_Damage_{i+1}.csv", mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        headers = ['Energia'] + [f'Material_{i+1}' for i in range(8)] + [f'Angulo_{i+1}' for i in range(8)] + [f'e_{j+1}' for j in range(120000)]
        csv_writer.writerow(headers)
        csv_writer.writerows(hist_csv)

print("Archivos generados exitosamente.")
