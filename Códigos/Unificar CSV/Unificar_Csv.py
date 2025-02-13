import pandas as pd
from tkinter import Tk, filedialog

def seleccionar_archivo(titulo):
    """
    Abre un cuadro de diálogo para seleccionar un archivo CSV.

    Args:
        titulo (str): Título del cuadro de diálogo.

    Returns:
        str: Ruta del archivo seleccionado.
    """
    root = Tk()
    root.withdraw()  # Oculta la ventana principal
    root.attributes('-topmost', True)  # Coloca el cuadro de diálogo al frente
    archivo_seleccionado = filedialog.askopenfilename(title=titulo, filetypes=[("Archivos CSV", "*.csv")])
    return archivo_seleccionado

def seleccionar_carpeta():
    """
    Abre un cuadro de diálogo para seleccionar una carpeta de destino.

    Returns:
        str: Ruta de la carpeta seleccionada.
    """
    root = Tk()
    root.withdraw()  # Oculta la ventana principal
    root.attributes('-topmost', True)  # Coloca el cuadro de diálogo al frente
    carpeta_seleccionada = filedialog.askdirectory(title="Selecciona la carpeta de destino")
    return carpeta_seleccionada

def verificar_y_unir_csv(archivos, archivo_salida):
    """
    Verifica que todos los archivos CSV indicados tengan los mismos encabezados
    y combina sus filas en un solo archivo CSV.

    Args:
        archivos (list): Lista de rutas a los archivos CSV.
        archivo_salida (str): Ruta completa del archivo de salida combinado.

    Returns:
        None
    """
    if not archivos:
        print("No se seleccionaron archivos CSV.")
        return

    # Leer el primer archivo para obtener los encabezados
    encabezados = None
    with open(archivos[0], 'r') as f:
        encabezados = f.readline().strip()

    with open(archivo_salida, 'w') as salida:
        salida.write(encabezados + '\n')  # Escribe los encabezados en el archivo final

        for archivo in archivos:
            with open(archivo, 'r') as f:
                encabezados_actuales = f.readline().strip()
                if encabezados_actuales != encabezados:
                    print(f"El archivo {archivo} no tiene los mismos encabezados. No se unirá.")
                    continue
                for linea in f:
                    salida.write(linea)

    print(f"Archivos combinados y guardados en {archivo_salida}")

# Ejemplo de uso
if __name__ == "__main__":
    print("¿Cuántos archivos CSV deseas combinar?")
    cantidad = int(input("Ingresa la cantidad de archivos: "))

    archivos = []
    for i in range(cantidad):
        titulo = f"Selecciona el archivo CSV {i + 1}"
        archivo = seleccionar_archivo(titulo)
        if archivo:
            archivos.append(archivo)

    if archivos:
        carpeta_salida = seleccionar_carpeta()
        if carpeta_salida:
            nombre_salida = input("Ingresa el nombre del archivo de salida (con extensión .csv): ")
            ruta_salida = f"{carpeta_salida}/{nombre_salida}"
            verificar_y_unir_csv(archivos, ruta_salida)
        else:
            print("No se seleccionó una carpeta de destino.")
    else:
        print("No se seleccionaron archivos suficientes para combinar.")
