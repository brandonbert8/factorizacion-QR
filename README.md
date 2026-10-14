# QR Numerical Lab

Aplicación educativa de **Métodos Numéricos II** para estudiar la factorización QR y resolver problemas reales de mínimos cuadrados con implementaciones propias de **Gram-Schmidt modificado** y **Householder**.

## Objetivos académicos

- Relacionar `A = QR` y `Rβ = Qᵀy` con regresión, sensibilidad numérica y reconstrucción de trayectorias.
- Inspeccionar proyecciones y reflectores calculados en cada iteración.
- Distinguir el error del algoritmo del condicionamiento del problema.
- Comparar resultados con SciPy como referencia independiente.

## Tecnologías

Python 3.12+, Streamlit, Plotly, NumPy, SciPy, SymPy, pandas y Pytest. SymPy representa el modelo de regresión calculado; los valores numéricos proceden de los algoritmos propios.

## Instalación y ejecución

En Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

En Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Abre la URL local que muestra Streamlit. Usa la barra lateral de Streamlit para navegar entre laboratorios. El tema inicial es oscuro; cambia a claro desde **Apariencia y movimiento** en un laboratorio o desde la barra lateral de la portada.

Cada ejercicio tiene una única presentación guiada. La sección **Resolución con QR** es el centro: muestra animaciones vectoriales de cada proyección de Gram-Schmidt o reflector de Householder, los productos de Qᵀy y la sustitución regresiva hasta obtener los coeficientes. Las matrices se consultan como comprobación. Las observaciones breves junto a las escenas y operaciones explican su significado matemático. En **Apariencia y movimiento** puedes cambiar tema o reducir movimiento.

## QR dentro de los tres ejemplos

En **Resolución con QR** de cada laboratorio se usa la matriz y el vector de observaciones del propio ejercicio:

- **Paso a paso:** animaciones de normalización, proyecciones y reflectores, seguidas de Qᵀy y sustitución regresiva. Cambia de método, avanza, retrocede, reinicia o salta a una operación.
- **Comparar métodos:** animación por iteraciones de ambos métodos sobre la misma A. MGS muestra Q, R y residuos pendientes; Householder muestra la transformación por reflectores y la reconstrucción de A. Se comparan errores QR, ortogonalidad, RMSE, coeficientes y el número de proyecciones y reflectores del ejercicio.
- **Reconstruir A:** suma animada de los términos qᵢrᵢⱼ para recuperar una columna de A, comparando ambos métodos y usando todas las componentes. La triangularidad de R determina cuántos términos hacen falta.
- **Precisión y sensibilidad:** animación de QₖᵀQₖ−I durante la construcción de las columnas, comprobación con QR económica y mínimos cuadrados por QR con pivoteo de SciPy, comparación de factores con signos unificados y respuesta animada de los coeficientes a perturbaciones controladas de y.

Para matrices largas, la animación amplía las primeras seis filas para que sean legibles; todos los cálculos usan los datos completos y los factores finales completos se pueden consultar. En proyectiles, la comparación respeta la ecuación seleccionada y el tratamiento de la gravedad. El modo de movimiento reducido muestra los resultados sin reproducir la transición.

Las ayudas están integradas como texto discreto junto a las gráficas y fórmulas, sin botones de guion o de notas de presentación.

Las reflexiones son discretas: el movimiento ilustra la resta y la conservación de norma se comprueba entre los extremos. Cada avance de la comparación es una iteración, no una medición de rendimiento. El estudio de sensibilidad mantiene A fija, usa una dirección singular sensible y compara cada solución perturbada con la solución base de su propio método. La diferencia respecto de SciPy es concordancia numérica, no un error frente a una verdad exacta.

## Los tres laboratorios

1. **Regresión inmobiliaria:** datos sintéticos identificados, modelo lineal o cuadrático, predicción, residuos y comparación de QR.
2. **Estabilidad numérica:** matriz casi dependiente `Aε`, barridos de ε y ruido, métricas de reconstrucción, ortogonalidad, condición y error de solución. La referencia SciPy solo valida los coeficientes.
3. **Projectile Mission Control:** simulación con mediciones ruidosas, ajuste de coordenadas, gravedad fija o estimada, parámetros físicos y residuos.


## Convenciones numéricas

- Para `A` de dimensión `m × n`, `m ≥ n`, las factorizaciones son económicas: `Q` es `m × n` y `R` es `n × n`.
- Gram-Schmidt modificado usa una pasada y diagonal positiva de `R`. Householder escoge el signo del reflector para evitar cancelación; su diagonal de `R` puede ser negativa.
- Se detectan columnas con norma residual menor o igual a `ε_machine · max(m,n) · ||A||₂`. Una condición mayor que `1/√ε_machine` produce advertencia de sensibilidad.
- El sistema triangular se resuelve mediante sustitución regresiva explícita. No se usan ecuaciones normales.
- La gravedad conocida en el módulo de proyectiles fija `c₂ = −g/2` y ajusta `c₀,c₁` por QR; se calcula además un ajuste cuadrático libre para comparar.

## Estructura

```text
app.py                 Inicio y navegación
pages/                 Tres laboratorios Streamlit
numerical_methods/     QR, sustitución regresiva, mínimos cuadrados y métricas
experiments/           Datos simulados y configuraciones experimentales
components/            Diseño, matrices, pasos y exportaciones
tests/                 Validación matemática con Pytest
```

## Pruebas

```bash
python -m pytest -q
```

Las tolerancias de los tests dependen de los datos usados. Para matrices casi singulares, el laboratorio muestra explícitamente la sensibilidad en lugar de exigir precisión uniforme.
