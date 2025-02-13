import os
import re
import math
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

# Función para leer el archivo .k
def leer_archivo_k(ruta_archivo):
    with open(ruta_archivo, 'r') as archivo:
        contenido = archivo.readlines()
    return contenido

# Función para calcular coseno y seno
def calcular_trigonometria(angulos):
    resultados = {}
    for i, angulo in enumerate(angulos, start=1):
        radianes = math.radians(angulo)  # Convertir ángulo de grados a radianes
        resultados[f"cosply{i}"] = f"{math.cos(radianes):.7f}"
        resultados[f"sinply{i}"] = f"{math.sin(radianes):.7f}"
        resultados[f"sinplyneg{i}"] = f"{math.sin(radianes)*-1:.7f}"
    return resultados

# Función para modificar el contenido de un archivo línea por línea usando un patrón precompilado
def modificar_contenido_bloques(ruta_entrada, ruta_salida, cambios):
    # Crear una expresión regular que busque todas las variables a reemplazar
    patron = re.compile(r'\b(' + '|'.join(re.escape(k) for k in cambios.keys()) + r')\b')
    
    # Función que retorna el valor correspondiente para cada variable encontrada
    def reemplazar(match):
        return cambios[match.group(0)]
    
    # Leer el archivo de entrada y escribir el archivo modificado
    with open(ruta_entrada, 'r') as entrada, open(ruta_salida, 'w') as salida:
        for linea in entrada:
            salida.write(patron.sub(reemplazar, linea))

# Función para validar la entrada de la configuración de laminado
def validar_configuracion_laminado(configuracion):
    configuracion = configuracion.strip()
    if configuracion.startswith("[") and configuracion.endswith("]"):
        configuracion = configuracion[1:-1].split(',')
        if len(configuracion) == 8:
            for valor in configuracion:
                valor = valor.strip()
                if valor not in ['CF', 'GF']:
                    return None
            return configuracion
    return None

# Función para validar la entrada de ángulos
def validar_angulos(angulos):
    angulos = angulos.strip()
    if angulos.startswith("[") and angulos.endswith("]"):
        angulos = angulos[1:-1].split(',')
        if len(angulos) == 8:
            try:
                return [float(angulo.strip()) for angulo in angulos]
            except ValueError:
                return None
    return None

# Función principal
def main():
    # Ocultar la ventana de Tkinter
    Tk().withdraw()

    # Seleccionar el archivo .k original
    print("Seleccione el archivo .k original:")
    ruta_archivo = askopenfilename(filetypes=[("Archivos .k", "*.k")])
    if not ruta_archivo:
        print("No se seleccionó un archivo .k. Saliendo...")
        return

    # Seleccionar la carpeta destino
    print("Seleccione la carpeta destino:")
    carpeta_destino = askdirectory()
    if not carpeta_destino:
        print("No se seleccionó una carpeta destino. Saliendo...")
        return

    # Seleccionar el archivo de configuraciones
    print("Seleccione el archivo de configuraciones:")
    ruta_configuraciones = askopenfilename(filetypes=[("Archivos de texto", "*.txt")])
    if not ruta_configuraciones:
        print("No se seleccionó un archivo de configuraciones. Saliendo...")
        return

    # Leer el archivo de configuraciones
    with open(ruta_configuraciones, 'r') as archivo:
        configuraciones = archivo.readlines()

    # Calcular el total de configuraciones
    total_configuraciones = len(configuraciones)
    print(f"Total de configuraciones a procesar: {total_configuraciones}")

    # Solicitar al usuario la energía inicial del impacto LVI
    while True:
        try:
            energy = float(input("Ingrese la energía inicial del impacto LVI en Joules: "))
            energy_mostrada = int(energy)
            mass = 2.2173
            in_velocity = math.sqrt((2 * energy) / mass) * -1
            in_velocity = f"{in_velocity:.2f}"
            print(f"La velocidad inicial es de: {in_velocity} m/s")
            break
        except ValueError:
            print("Entrada no válida. Por favor, ingrese un número con 2 decimales.")

    # Solicitar al usuario el número de carpetas a crear
    while True:
        try:
            num_carpetas = int(input("Ingrese el número de subcarpetas para distribuir los archivos: "))
            if num_carpetas > 0:
                break
            else:
                print("El número de subcarpetas debe ser mayor a cero.")
        except ValueError:
            print("Entrada no válida. Por favor, ingrese un número entero.")

    # Crear las subcarpetas y dividir las configuraciones
    subcarpetas = [os.path.join(carpeta_destino, f"Subcarpeta_{i+1}") for i in range(num_carpetas)]
    for subcarpeta in subcarpetas:
        os.makedirs(subcarpeta, exist_ok=True)

    configuraciones_por_carpeta = total_configuraciones // num_carpetas
    configuraciones_restantes = total_configuraciones % num_carpetas

    distribucion_configuraciones = []
    inicio = 0
    for i in range(num_carpetas):
        fin = inicio + configuraciones_por_carpeta + (1 if i < configuraciones_restantes else 0)
        distribucion_configuraciones.append(configuraciones[inicio:fin])
        inicio = fin

    # Procesar cada configuración y distribuir en subcarpetas
    for i, subcarpeta in enumerate(subcarpetas):
        configuraciones_actuales = distribucion_configuraciones[i]
        archivo_txt_subcarpeta = os.path.join(subcarpeta, f"Subcarpeta_{i+1}.txt")

        with open(archivo_txt_subcarpeta, 'w') as archivo_txt:
            archivo_txt.writelines(configuraciones_actuales)

        for j, configuracion_linea in enumerate(configuraciones_actuales):
            laminado_str, angulos_str = configuracion_linea.split(' ', 1)

            configuracion = validar_configuracion_laminado(laminado_str)
            angulos = validar_angulos(angulos_str)

            if not configuracion or not angulos:
                print(f"Configuración inválida en subcarpeta {i+1}: {configuracion_linea.strip()}")
                continue

            cambios = {"invelocity": in_velocity}
            for k, material in enumerate(configuracion):
                cambios[f"matply{k+1}"] = "1" if material == "CF" else "2"

            trigonometria = calcular_trigonometria(angulos)
            cambios.update(trigonometria)

            nombre_nuevo = f"{energy_mostrada}J_" + "_".join(configuracion) + "_" + "_".join(map(str, map(int, angulos)))
            subcarpeta_archivo = os.path.join(subcarpeta, nombre_nuevo)
            os.makedirs(subcarpeta_archivo, exist_ok=True)

            ruta_archivo_modificado = os.path.join(subcarpeta_archivo, f"{nombre_nuevo}.k")

            modificar_contenido_bloques(ruta_archivo, ruta_archivo_modificado, cambios)

            print(f"Archivo generado: {ruta_archivo_modificado}")

    print("Distribución y generación de archivos completada.")

if __name__ == "__main__":
    main()
