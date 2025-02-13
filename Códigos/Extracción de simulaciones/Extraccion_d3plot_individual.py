from ansys.dpf import core as dpf
import numpy as np
import matplotlib.pyplot as plt

model = dpf.Model(r"C:\DAVID\Prueba_time\d3plot")
print(model)

# Definimos rangos de tiempo (por ejemplo de 1 a 100)
itime_values = range(1, 101)

# Listas para almacenar los datos de daños
damages_1_list = []
damages_2_list = []
damages_3_list = []
damages_4_list = []
damages_5_list = []

for itime in itime_values:
    # Extraemos los resultados para el itime específico
    hist = getattr(model.results, "history_variablesihv__[1__5]").on_time_scoping([itime]).eval()
    disp = getattr(model.results, "displacement").on_time_scoping([itime]).eval()
    
    # Históricas (asumiendo las mismas posiciones que en tu código original)
    hist1 = hist[0].data
    hist2 = hist[1].data
    hist3 = hist[2].data
    hist4 = hist[3].data
    hist5 = hist[4].data
    


    # Tomamos los primeros 120000 valores (esto depende de tus datos)
    composite_histvar1 = hist1[:120000]
    composite_histvar2 = hist2[:120000]
    composite_histvar3 = hist3[:120000]
    composite_histvar4 = hist4[:120000]
    composite_histvar5 = hist5[:120000]
    
    # Calculamos daños (ejemplo como en tu código)
    damages_1 = 120000 - sum(composite_histvar1)
    damages_2 = 120000 - sum(composite_histvar2)
    damages_3 = 120000 - sum(composite_histvar3)
    damages_4 = 120000 - sum(composite_histvar4)
    damages_5 = 120000 - sum(composite_histvar5)
    
    # Agregamos a las listas
    damages_1_list.append(damages_1)
    damages_2_list.append(damages_2)
    damages_3_list.append(damages_3)
    damages_4_list.append(damages_4)
    damages_5_list.append(damages_5)

# Ahora generamos las gráficas
plt.figure(figsize=(10, 8))

# Graficamos cada damages_x vs itime
plt.plot(itime_values, damages_1_list, label='Damages 1')
plt.plot(itime_values, damages_2_list, label='Damages 2')
plt.plot(itime_values, damages_3_list, label='Damages 3')
plt.plot(itime_values, damages_4_list, label='Damages 4')
plt.plot(itime_values, damages_5_list, label='Damages 5')

plt.xlabel('Tiempo (itime)')
plt.ylabel('Daños (Sum)')
plt.title('Evolución de daños a lo largo del tiempo')
plt.legend()
plt.grid(True)
plt.show()

print(f"Cantidad de elementos en damages_1_list: {len(hist1)}")
print(f"Cantidad de elementos en damages_2_list: {len(hist2)}")
print(f"Cantidad de elementos en damages_3_list: {len(hist3)}")
print(f"Cantidad de elementos en damages_4_list: {len(hist4)}")
print(f"Cantidad de elementos en damages_5_list: {len(hist5)}")

print(f"Cantidad de elementos en : {damages_1}")
print(f"Cantidad de elementos en : {damages_2}")
print(f"Cantidad de elementos en : {damages_3}")
print(f"Cantidad de elementos en : {damages_4}")
print(f"Cantidad de elementos en : {damages_5}")