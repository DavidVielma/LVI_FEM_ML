import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tkinter import Tk
from tkinter.filedialog import askdirectory

def get_short_path(long_path):
    """Obtiene la ruta corta de un archivo o carpeta."""
    try:
        return subprocess.check_output(f'for %I in ("{long_path}") do @echo %~sI', shell=True).decode().strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error al obtener la ruta corta: {e}")

def detect_energy_from_folder(folder_name):
    """Detecta la energía de impacto a partir de los primeros dos dígitos de una carpeta."""
    match = re.match(r'^(\d{2})', folder_name)
    if match:
        return int(match.group(1))
    return None

def validate_energy(folder_name, detected_energy):
    """Valida con el usuario si la energía detectada es correcta."""
    print(f"Energía detectada de la carpeta '{folder_name}': {detected_energy}J")
    user_response = input("¿Es correcta esta energía? (s/n): ").strip().lower()
    if user_response == 's':
        return detected_energy
    else:
        while True:
            try:
                new_energy = int(input("Ingrese la energía correcta en Julios: "))
                return new_energy
            except ValueError:
                print("Entrada no válida. Por favor, ingrese un número entero.")

def get_folders_and_files(base_dir):
    """Obtiene las subcarpetas y los archivos necesarios en cada subcarpeta."""
    try:
        subcarpetas = [os.path.join(base_dir, sub) for sub in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, sub))]
        subcarpetas.sort()  # Ordenar para mantener consistencia
        configurations = []
        for subcarpeta in subcarpetas:
            archivo_txt = os.path.join(subcarpeta, f"{os.path.basename(subcarpeta)}.txt")
            if not os.path.exists(archivo_txt):
                raise FileNotFoundError(f"Archivo de configuraciones no encontrado: {archivo_txt}")
            # Detectar energía desde la primera carpeta dentro de la subcarpeta
            primera_carpeta = next((f for f in os.listdir(subcarpeta) if os.path.isdir(os.path.join(subcarpeta, f))), None)
            if not primera_carpeta:
                raise FileNotFoundError(f"No se encontraron carpetas dentro de: {subcarpeta}")
            energia_detectada = detect_energy_from_folder(primera_carpeta)
            energia_final = validate_energy(primera_carpeta, energia_detectada)
            configurations.append((subcarpeta, energia_final, archivo_txt))
        return configurations
    except Exception as e:
        raise RuntimeError(f"Error al obtener carpetas y archivos: {e}")

def run_simulations(base_dir, energy, input_file):
    """Ejecuta las simulaciones para una carpeta específica."""
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        raise RuntimeError(f"Error al leer el archivo de configuraciones '{input_file}': {e}")

    iteration = 0
    total_iterations = len(lines)

    for line in lines:
        # Usar expresiones regulares para extraer los materiales y dimensiones entre corchetes
        match = re.match(r'\[(.*?)\] \[(.*?)\]', line.strip())
        if match:
            materials_str = match.group(1)
            dimensions_str = match.group(2)

            materials = materials_str.split(',')
            dimensions = dimensions_str.split(',')

            folder_name = f"{energy}J_" + "_".join(materials + dimensions)
            folder_path = os.path.join(base_dir, folder_name)

            k_file_name = f"{folder_name}.k"
            k_file_path = os.path.join(folder_path, k_file_name)

            try:
                short_k_file_path = get_short_path(k_file_path)
            except RuntimeError as e:
                print(f"Error al obtener la ruta corta para '{k_file_path}': {e}")
                continue

            print(f"Procesando carpeta: {folder_name}, archivo .k: {k_file_name}")
            print(f"Ruta corta del archivo .k: {short_k_file_path}")

            try:
                os.chdir(folder_path)
            except FileNotFoundError:
                print(f"Carpeta no encontrada: {folder_path}. Saltando...")
                continue

            command = r'call "C:\Program Files\ANSYS Inc\v231\ansys\bin\winx64\lsprepost48\LS-Run 1.0\lsdynamsvar.bat" && ' \
                      r'"C:\Program Files\ANSYS Inc\v231\ansys\bin\winx64\lsdyna_dp.exe"'

            try:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
                parameters = f"i={short_k_file_path} ncpu=4 memory=2048m s=fname\n"
                process.stdin.write(parameters)
                process.stdin.flush()
                stdout, stderr = process.communicate()

                print(stdout)
                if stderr:
                    print(f"Errores: {stderr}")
            except Exception as e:
                print(f"Error al ejecutar la simulación: {e}")
                continue

            iteration += 1
            print(f"Simulación {iteration}/{total_iterations} completada")

    print(f"Total de simulaciones realizadas en '{base_dir}': {iteration}")

def main():
    # Seleccionar la carpeta madre mediante el navegador de carpetas
    Tk().withdraw()
    print("Seleccione la carpeta madre donde están las subcarpetas específicas:")
    base_dir = askdirectory()
    if not base_dir or not os.path.isdir(base_dir):
        print("La carpeta madre no es válida. Saliendo...")
        return

    try:
        configurations = get_folders_and_files(base_dir)
    except RuntimeError as e:
        print(f"Error al detectar carpetas y archivos: {e}")
        return

    num_terminales = len(configurations)
    print(f"Se detectaron {num_terminales} subcarpetas para procesar.")

    with ProcessPoolExecutor(max_workers=num_terminales) as executor:
        futures = []
        for subcarpeta, energy, input_file in configurations:
            futures.append(executor.submit(run_simulations, subcarpeta, energy, input_file))

        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error en una de las simulaciones: {e}")

    print("\nTodas las simulaciones han sido completadas.")

if __name__ == "__main__":
    main()
