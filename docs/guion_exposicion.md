# Guion de exposición — QR Numerical Lab (8 minutos)

## Antes de empezar

Ejecuta `streamlit run app.py`. La portada ofrece los tres ejemplos: regresión inmobiliaria, estabilidad numérica y proyectiles. En cada uno, **Resolver con QR →** abre directamente su resolución. Allí puedes elegir **Paso a paso** o **Comparar métodos**, siempre con los datos del mismo ejercicio.

Las animaciones comienzan pausadas. Usa **Animar operación**, **Paso →**, el selector de operaciones y **Reiniciar**. **Reducir movimiento** muestra el resultado directamente.

## 0:00–0:40 · La idea común

Presenta `A = QR`: Q tiene columnas ortonormales y R es triangular superior. Para `m ≥ n`, Q económica es `m×n` y R es `n×n`. Después se resuelve `Rβ = Qᵀy` para obtener los coeficientes del ejercicio.

**Di:** «En los tres ejemplos vamos a observar cómo Gram-Schmidt modificado y Householder factorizan la matriz de sus datos y cómo comparar la precisión de ambos».

## 0:40–3:20 · Regresión inmobiliaria: explicar los dos métodos

1. En **Experimento**, muestra las viviendas y el modelo lineal o cuadrático. Cada fila de A corresponde a una vivienda; las columnas son 1, superficie y, si corresponde, superficie².
2. Pulsa **Resolver con QR →**, selecciona **Gram-Schmidt modificado** y conserva **Paso a paso**. Anima la normalización: la norma se guarda en R y la dirección unitaria se incorpora a Q.
3. Avanza a una proyección: el vector amarillo es la componente que se resta. El residuo queda perpendicular a q. Señala la descomposición y la comprobación del producto escalar. La variante modificada usa los residuos actualizados.
4. Cambia a **Householder** sobre los mismos datos. Muestra la normal v del espejo y anima la reflexión de la columna objetivo hasta el eje e₁ del bloque activo. Se anulan entradas bajo la diagonal; el mismo espejo se aplica a las demás columnas pendientes.
5. En **Comparar métodos**, reproduce la construcción de las matrices. MGS cumple `A = QₖRₖ + V_pendiente`; Householder cumple `Bₖ = Hₖ…H₁A` y `A = QₖBₖ`. Al terminar, V es cero y B tiene R sobre un bloque de ceros.
6. Compara error QR, ortogonalidad y los coeficientes del precio. Regresa a **Paso a paso** y salta a Qᵀy o a la sustitución regresiva para relacionar la factorización con el modelo obtenido.

**Aclaraciones:** la reflexión es discreta; el movimiento ilustra una resta y la norma se conserva entre los extremos. Los signos de Q y R pueden diferir y seguir dando el mismo producto QR. Cada avance de la comparación es una iteración, no una medición de velocidad.

## 3:20–5:30 · Estabilidad numérica: comparar precisión

1. En **Experimento**, acerca las columnas de Aε disminuyendo ε y cambia el ruido de las observaciones.
2. En **Resolución con QR**, observa una proyección de MGS: un residuo pequeño es lo que debe normalizarse. La ampliación del residuo está identificada en la gráfica.
3. Cambia a **Comparar métodos**. Ambas construcciones usan exactamente la Aε y las observaciones perturbadas del experimento. Compara errores de reconstrucción y pérdida de ortogonalidad.
4. Householder suele conservar mejor la ortogonalidad cuando las columnas casi dependen entre sí. El error de los coeficientes también depende del condicionamiento y del ruido.

**Di:** «Estabilidad describe el algoritmo; condicionamiento describe el problema. Una buena QR no elimina la sensibilidad de A».

## 5:30–7:30 · Proyectiles: QR en las ecuaciones del movimiento

1. Muestra brevemente la trayectoria reconstruida. En **Resolución con QR**, elige **Altura y(t)**: A contiene 1, t y t² si estimamos g.
2. Anima una operación de ambos métodos y abre **Comparar métodos**. Compara los factores y los coeficientes de altura inicial, velocidad vertical y −g/2.
3. Selecciona **Posición horizontal x(t)**. La misma sección ahora factoriza `[1,t]` y compara los coeficientes de posición inicial y velocidad horizontal.
4. Si usas **g conocida**, aclara que se factoriza `[1,t]` y se ajusta `y + gt²/2`; g se fija, no se estima. La comparación utiliza ese mismo sistema transformado.

Para matrices largas se muestran las primeras seis filas ampliadas. Los cálculos usan todas las mediciones; los factores finales completos se consultan en los desplegables.

## 7:30–8:00 · Cierre

**Di:** «Los tres ejemplos comparten la factorización QR. Gram-Schmidt construye direcciones ortonormales mediante proyecciones; Householder crea una triangular mediante reflectores. Después usamos sus factores para resolver el sistema de cada ejercicio. Reconstrucción, ortogonalidad, ajuste y sensibilidad miden aspectos distintos».

Si preguntan por validación, menciona las pruebas de estados intermedios, geometría de las animaciones, reconstrucción, ortogonalidad, mínimos cuadrados y comparación con SciPy.
