import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, confusion_matrix
import xgboost as xgb

from scipy.signal import savgol_filter

# --------------------------------------------------------------------
# 1. CONFIGURACIÓN DE DIRECTORIO DE SALIDA
# --------------------------------------------------------------------
output_dir = r"C:\Users\peped\OneDrive - usach.cl\Univeridad\14._Simulaciones Tesis\Simulaciones\0.-Casos Tesis\Extraccion\ML\Resultados_XGBoost"

# Crear carpeta si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --------------------------------------------------------------------
# 2. CARGA DE DATOS
# --------------------------------------------------------------------
train_data = pd.read_csv(r"C:\Users\peped\OneDrive - usach.cl\Univeridad\14._Simulaciones Tesis\Simulaciones\0.-Casos Tesis\Extraccion\2400 Casos Unificado\2400_Unificado.csv")
valid_data = pd.read_csv(r"C:\Users\peped\OneDrive - usach.cl\Univeridad\14._Simulaciones Tesis\Simulaciones\0.-Casos Tesis\Extraccion\150 Casos Validacion\Salida_150Sim_Valid.csv")

# Transformar CF/GF a 0/1
cols_cf_gf = list(range(1, 9))
for df in [train_data, valid_data]:
    df.iloc[:, cols_cf_gf] = df.iloc[:, cols_cf_gf].replace({"CF": 0, "GF": 1}).astype(int)

# Convertir cualquier columna 'object' restante a 'category' y luego a numérica
for df in [train_data, valid_data]:
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes  

# --------------------------------------------------------------------
# 3. SEPARAR FEATURES (X) Y TARGET (Y)
# --------------------------------------------------------------------
X = train_data.iloc[:, 1:17]                   # Columnas de entrada
Y = train_data.iloc[:, [20, 21]].astype(float) # Damage_3 y Damage_4

# División train/test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42
)

# Conjunto de validación final
X_valid = valid_data.iloc[:, 1:17]
Y_valid = valid_data.iloc[:, [20, 21]].astype(float)

# --------------------------------------------------------------------
# 4. DEFINIR HIPERPARÁMETROS
# --------------------------------------------------------------------
param_grid = {
    "Damage_3": {
        'learning_rate': [0.06555053],
        'n_estimators': [990, 1000, 1010],
        'max_depth': [4, 5],
        'nthread': [1]
    },
    "Damage_4": {
        'learning_rate': [0.0501], #5530
        'n_estimators': [550, 580],
        'max_depth': [3, 5],
        'nthread': [1]
    }
}

# --------------------------------------------------------------------
# PARÁMETROS ADICIONALES
# --------------------------------------------------------------------
valor_Filtro = 30         # Porcentaje máximo de error para filtrado (ejemplo)
tamano_intervalo = 200    # <<--- AQUÍ DEFINES EL TAMAÑO DE LOS INTERVALOS
                          #      para las gráficas de barras

# Para almacenar resultados y un texto resumen
results = {}
summary_lines = []  # iremos guardando líneas de texto para el resumen

# --------------------------------------------------------------------
# 5. ENTRENAMIENTO Y EVALUACIÓN
# --------------------------------------------------------------------
for col in Y_train.columns:
    print(f"\nEntrenando modelo para {col}...\n")
    
    # GridSearchCV
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')
    grid = GridSearchCV(
        xgb_reg,
        param_grid[col],
        cv=3,
        scoring='neg_mean_absolute_error',
        error_score='raise'
    )
    grid.fit(X_train, Y_train[col])
    best_params = grid.best_params_

    # Modelo final
    final_xgb = xgb.XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        eval_metric="mae"
    )
    final_xgb.fit(X_train, Y_train[col],
                  eval_set=[(X_test, Y_test[col])],
                  verbose=True)

    # Predicciones en validación
    y_preds = final_xgb.predict(X_valid)

    # Error absoluto y percentil
    errores_abs = np.abs(Y_valid[col] - y_preds)
    percentil_alto = np.percentile(errores_abs, 98)

    # Error relativo y filtrado
    error_relativo = np.abs((Y_valid[col] - y_preds) / Y_valid[col]) * 100
    filtro = error_relativo <= valor_Filtro
    y_preds_filtrados = y_preds[filtro]
    Y_valid_filtrados = Y_valid[col][filtro]

    # R² unfiltered y filtered
    r2_original = r2_score(Y_valid[col], y_preds)
    r2_filtrado = (r2_score(Y_valid_filtrados, y_preds_filtrados) 
                   if len(y_preds_filtrados) > 0 else np.nan)

    # Nuevo percentil para datos filtrados
    errores_abs_filtrados = np.abs(Y_valid_filtrados - y_preds_filtrados)
    nuevo_percentil_alto = (np.percentile(errores_abs_filtrados, 98) 
                            if len(errores_abs_filtrados) > 0 else np.nan)

    # Guardar resultados
    results[col] = {
        "r2_original": r2_original,
        "r2_filtrado": r2_filtrado,
        "mejores_hiperparametros": best_params,
        "intervalo_96_original": percentil_alto,
        "intervalo_96_filtrado": nuevo_percentil_alto,
        "datos_filtrados": len(y_preds_filtrados),
        "predicciones": y_preds
    }

    # ==================================================
    #           G R Á F I C A S   ( y guardado )
    # ==================================================

    # A) Real vs. Predicho (unfiltered)
    plt.figure(figsize=(7, 5))
    idx_unf = np.argsort(Y_valid[col].values)
    X_real_unf = Y_valid[col].values[idx_unf]  # Real ordenado
    Y_pred_unf = y_preds[idx_unf]             # Predicho ordenado

    # Suavizado de la curva
    window_size = 11  # Ajustar según datos (debe ser impar)
    poly_order = 2
    if len(Y_pred_unf) >= window_size:
        Y_pred_unf_smooth = savgol_filter(Y_pred_unf, window_length=window_size, polyorder=poly_order)
    else:
        Y_pred_unf_smooth = Y_pred_unf

    Y_inf_unf_smooth = Y_pred_unf_smooth - percentil_alto
    Y_sup_unf_smooth = Y_pred_unf_smooth + percentil_alto

    plt.fill_between(
        X_real_unf,
        Y_inf_unf_smooth,
        Y_sup_unf_smooth,
        color='blue',
        alpha=0.15,
        edgecolor='none',
        linewidth=0,
        label=f'Intervalo ±{percentil_alto:.2f}'
    )
    plt.scatter(X_real_unf, Y_pred_unf, color='blue', alpha=0.6, s=20, label='Predicciones')

    min_val = min(X_real_unf.min(), Y_pred_unf.min())
    max_val = max(X_real_unf.max(), Y_pred_unf.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')

    plt.title(f"[{col}] Real vs. Predicho (XGBoost) \nR² = {r2_original:.4f}")
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Guardar la figura
    file_unf = os.path.join(output_dir, f"{col}_Unfiltered_Real_vs_Pred.png")
    plt.savefig(file_unf, dpi=300)
    plt.show()

    # B) Real vs. Predicho (filtered)
    if len(Y_valid_filtrados) > 0:
        plt.figure(figsize=(7, 5))
        idx_filt = np.argsort(Y_valid_filtrados.values)
        X_real_filt = Y_valid_filtrados.values[idx_filt]
        Y_pred_filt = y_preds_filtrados[idx_filt]

        if len(Y_pred_filt) >= window_size:
            Y_pred_filt_smooth = savgol_filter(Y_pred_filt, window_size, poly_order)
        else:
            Y_pred_filt_smooth = Y_pred_filt

        if not np.isnan(nuevo_percentil_alto):
            Y_inf_filt_smooth = Y_pred_filt_smooth - nuevo_percentil_alto
            Y_sup_filt_smooth = Y_pred_filt_smooth + nuevo_percentil_alto
        else:
            Y_inf_filt_smooth = Y_pred_filt_smooth
            Y_sup_filt_smooth = Y_pred_filt_smooth

        plt.fill_between(
            X_real_filt,
            Y_inf_filt_smooth,
            Y_sup_filt_smooth,
            color='blue',
            alpha=0.15,
            edgecolor='none',
            linewidth=0,
            label=f'Intervalo ±{nuevo_percentil_alto:.2f}' if not np.isnan(nuevo_percentil_alto) else 'Intervalo'
        )
        plt.scatter(X_real_filt, Y_pred_filt, color='green', alpha=0.6, s=20, label='Predicciones (filt)')

        min_val_f = min(X_real_filt.min(), Y_pred_filt.min())
        max_val_f = max(X_real_filt.max(), Y_pred_filt.max())
        plt.plot([min_val_f, max_val_f], [min_val_f, max_val_f], 'r--', label='y = x')

        plt.title(f"[{col}] Real vs. Predicho (Filtered)\nR² = {r2_filtrado:.4f}")
        plt.xlabel("Real (filtered)")
        plt.ylabel("Predicho (filtered)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Guardar
        file_filt = os.path.join(output_dir, f"{col}_Filtered_Real_vs_Pred.png")
        plt.savefig(file_filt, dpi=300)
        plt.show()

    # C) Error Relativo (barras) - Unfiltered
    valor_intervalo = 250
    bins = np.arange(Y_valid[col].min(), Y_valid[col].max() + valor_intervalo, valor_intervalo)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-1)]
    Y_valid_binned = pd.cut(Y_valid[col], bins=bins, labels=labels, include_lowest=True)

    error_percentual = np.abs((Y_valid[col] - y_preds) / Y_valid[col]) * 100
    error_promedio_por_intervalo = error_percentual.groupby(Y_valid_binned).mean()

    plt.figure(figsize=(8, 6))
    error_promedio_por_intervalo.plot(kind='bar', color='skyblue')
    plt.xlabel('Elementos dañados')
    plt.ylabel('Promedio del Error (%)')
    plt.title(f'Promedio del Error por Intervalo de Daño (XGBoost) - {col}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    file_err_unf = os.path.join(output_dir, f"{col}_ErrorRel_Unfiltered.png")
    plt.savefig(file_err_unf, dpi=300)
    plt.show()

    # D) Error Relativo (barras) - Filtered
    if len(Y_valid_filtrados) > 0:
        Y_valid_filtrados_binned = pd.cut(Y_valid_filtrados, bins=bins, labels=labels, include_lowest=True)
        error_relativo_filtrado = np.abs((Y_valid_filtrados - y_preds_filtrados) / Y_valid_filtrados) * 100
        error_promedio_por_intervalo_filtrado = error_relativo_filtrado.groupby(Y_valid_filtrados_binned).mean()

        plt.figure(figsize=(8, 6))
        error_promedio_por_intervalo_filtrado.plot(kind='bar', color='skyblue')
        plt.xlabel('Elementos dañados')
        plt.ylabel('Promedio del Error (%)')
        plt.title(f'Promedio del Error por Intervalo de Daño (Filtered) - {col}')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()

        file_err_filt = os.path.join(output_dir, f"{col}_ErrorRel_Filtered.png")
        plt.savefig(file_err_filt, dpi=300)
        plt.show()

    # --------------------------------------------------------------------
    # 6. GRÁFICA DE BARRAS DE ERROR RELATIVO
    # --------------------------------------------------------------------
    # Calcular el error relativo para cada predicción
    error_relativo_total = np.abs((Y_valid[col] - results[col]["predicciones"]) / Y_valid[col]) * 100
    # Crear un DataFrame con los errores relativos
    error_data_total = pd.DataFrame({
        "Error Relativo (%)": error_relativo_total
    })
    # Agrupar los errores relativos en intervalos de 5%
    bins = np.arange(0, 105, 5)
    error_data_total['Intervalo'] = pd.cut(error_data_total["Error Relativo (%)"], bins, right=False)
    # Contar la cantidad de casos en cada intervalo
    error_counts = error_data_total['Intervalo'].value_counts().sort_index()
    # Calcular el porcentaje de casos en cada intervalo
    error_percentages = (error_counts / len(error_data_total)) * 100
    # Crear la gráfica de barras
    plt.figure(figsize=(8, 6))
    sns.barplot(x=error_percentages.index.astype(str), y=error_percentages.values, color="skyblue")
    plt.xlabel("Intervalo de Error Relativo (%)")
    plt.ylabel("Porcentaje de Casos (%)")
    plt.title(f"Distribución del Error Relativo en Intervalos de 5% (XGBoost)- {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Guardar la figura
    file_error_dist = os.path.join(output_dir, f"Distribucion_Error_Relativo_{col}.png")
    plt.savefig(file_error_dist, dpi=300)
    plt.show()


# --------------------------------------------------------------------
# 6. MATRIZ DE CORRELACIÓN (Incluye Damage_3 y Damage_4 en el train)
# --------------------------------------------------------------------
# Matriz de correlación con datos originales
X_train["Damage_3"] = Y_train["Damage_3"]
X_train["Damage_4"] = Y_train["Damage_4"]

plt.figure(figsize=(12, 10))
correlation_matrix_original = X_train.corr()
sns.heatmap(correlation_matrix_original, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación entre Variables de Entrada y Damage_3 / Damage_4 ")
plt.tight_layout()

file_corr_original = os.path.join(output_dir, "Correlation_Matrix_Original.png")
plt.savefig(file_corr_original, dpi=300)
plt.show()

# # Matriz de correlación con datos predecidos
# X_valid["Damage_3_Pred"] = results["Damage_3"]["predicciones"]
# X_valid["Damage_4_Pred"] = results["Damage_4"]["predicciones"]

# plt.figure(figsize=(12, 10))
# correlation_matrix_pred = X_valid.corr()
# sns.heatmap(correlation_matrix_pred, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title("Matriz de Correlación entre Variables de Entrada y Damage_3 / Damage_4 (Predicciones)")
# plt.tight_layout()

# file_corr_pred = os.path.join(output_dir, "Correlation_Matrix_Pred.png")
# plt.savefig(file_corr_pred, dpi=300)
# plt.show()


# --------------------------------------------------------------------
# 7. RESUMEN FINAL (y guardarlo en TXT)
# --------------------------------------------------------------------
summary_lines.append("\n🔹 RESUMEN FINAL 🔹")
for col, res in results.items():
    line = (
        f"\n📌 Variable: {col}\n"
        f"   - R² sin filtrar: {res['r2_original']:.4f}\n"
        f"   - R² con filtrado: {res['r2_filtrado']:.4f}\n"
        f"   - Mejores hiperparámetros: {res['mejores_hiperparametros']}\n"
        f"   - Intervalo 96% sin filtrar: ±{res['intervalo_96_original']:.4f}\n"
        f"   - Intervalo 96% con filtrado: ±{res['intervalo_96_filtrado']:.4f}\n"
        f"   - Datos dentro del filtro (error < {valor_Filtro}%): {res['datos_filtrados']}\n"
    )
    summary_lines.append(line)

summary_text = "\n".join(summary_lines)

# Mostrar en consola
print(summary_text)

# Guardar en txt
txt_file = os.path.join(output_dir, "Resumen_XGBoost.txt")
with open(txt_file, 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"\nResumen guardado en: {txt_file}")
print(f"Todas las gráficas guardadas en: {output_dir}")
