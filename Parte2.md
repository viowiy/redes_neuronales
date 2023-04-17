
# Desarrollo Teórico 

**Integrantes**
- Ricardo Ortega Bolaños.
- Viowi Y. Cabrisas Amuedo.

## Base de Datos y Problema

El problema a resolver trata sobre el cáncer de mama, uno de los tumores malignos más frecuentes en el mundo y su investigación constituye un desafío en la práctica médica. El diagnóstico precoz del cáncer de mama se ve obstaculizado por la heterogeneidad clínica y patológica de la enfermedad y las limitaciones de los métodos de cribado utilizados hasta el momento [[1]](http://scielo.sld.cu/pdf/rcim/v13n1/1684-1859-rcim-13-01-e385.pdf).

El presente ejercicio se orientó hacia la clasificación del cáncer de mama a partir de datos de estudios anatomopatológicos de una masa tumoral, en benigno (0) o maligno (1). La base de datos seleccionada para el mismo pertenece al “Machine Learning Repository” de la Universidad de Wisconsin y almacena información sobre un estudio morfométrico del núcleo celular, a partir de la toma de una muestra de biopsia por aspiración con aguja fina (BAAF) de una masa tumoral en la mama de los pacientes examinados.
Esta base de datos contiene 569 registros y 32 atributos, contando entre estos el ID del paciente, su clasificación y otros 30 atributos de valor real. Se encontraron 357 con diagnóstico de tumor benigno y 212 con diagnóstico de tumor maligno de la mama.[[2]](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

De esos 30, se calculan diez características de valor real para cada núcleo celular en el estudio de Wisconsin, las cuales serán utilizadas en el presente trabajo:

1. **Radio:** El radio de la célula fue medido promediando la longitud de los segmentos de líneas radiales definidos por el centroide de la célula y los puntos individuales en el límite de la célula. O sea, media de las distancias desde el centro a los puntos del perímetro).

<img src="https://github.com/viowiy/redes_neuronales/blob/main/images/morfometria_fig01.png?raw=true">

<center>Figura 1. Líneas radiales medidas en una célula.</center>

2. **Textura:** Fue medida encontrando la varianza en intensidades de escala de grises en los pixeles de la computadora.

<img src="https://github.com/viowiy/redes_neuronales/blob/main/images/morfometria_fig02.png?raw=true">

<center>Figura 2. Ejemplo de una imagen tomada por un sistema de visión por computadora y el contorno de la célula.</center>

3. **Perímetro:** Definido como la distancia total entre puntos individuales llamados puntos serpiente. Estos puntos individuales comprenden las líneas blancas en el perímetro de las células.

4. **Área:** Se obtiene contando el número de pixeles en el interior de la línea blanca añadiendo la mitad de los pixeles en el perímetro.

5. **Suavidad:** se calcula midiendo la diferencia entre la longitud de una línea radial y la longitud principal que la rodea. Básicamente, la suavidad es la variación local en las longitudes de radio.

6. **Compacidad:** El perímetro y el área son combinados para calcular la medida de compacidad; la cual es una medida de forma. La compacidad está dada por la fórmula: (perímetro^2 / área - 1.0).

7. **Concavidad:** Es la severidad de las porciones cóncavas del contorno. Analiza las irregularidades de forma en el núcleo de la célula. Se miden el número y la severidad de las concavidades y hendiduras en el núcleo de la célula. Se dibujan cuerdas entre cada punto blanco no adyacente y miden hasta qué punto el límite real del núcleo se encuentra en el interior de cada cuerda.

<img src="https://github.com/viowiy/redes_neuronales/blob/main/images/morfometria_fig03.png?raw=true">

<center>Figura 3. Cuerdas usadas para calcular la concavidad.</center>

8. **Puntos cóncavos:** Usan una medida similar a la concavidad, pero esta característica solo mide el número de puntos cóncavos del contorno, más que la magnitud, de las concavidades del contorno.

9. **Simetría:** Se obtiene encontrando la línea más larga que pase por el centro. Entonces, se trazan líneas perpendiculares a dicha línea para medir la diferencia de longitudes en las dos direcciones de la lineal central.

<img src="https://github.com/viowiy/redes_neuronales/blob/main/images/morfometria_fig04.png?raw=true" width="200">

<center>Figura 4. Segmentos usados en el cálculo de la simetría.</center>

10. **Dimensión fractal:** Es una característica de forma , es decir, a mayor valor corresponde a un menor contorno y por tanto a una mayor probabilidad malignidad. La dimensión fractal se aproxima usando la aproximación de costa de Mandelbrot ("aproximación de costa" - 1). El perímetro del núcleo es medido usando “reglas” cada vez más grandes. Esto es, a medida que aumenta el tamaño de la regla, decrece la precisión de la medición, el perímetro observado disminuye. Ahora, trazando estos valores a una escala logarítmica y medir la pendiente descendente da el negativo de una aproximación de la dimensión fractal.

<img src="https://github.com/viowiy/redes_neuronales/blob/main/images/morfometria_fig05.png?raw=true" widht="300">

<center>Figura 5. Secuencia de medidas para calcular dimensión fractal.</center>

Normalmente, la Selección de Subconjuntos de Características (FSS, por sus siglas en inglés: Feature Subset Selection) es usado para reducir la dimensionalidad, lo que significa que reduce el número de variables, atributos o características con las cuales se describen los objetos y encontrar su influencia en un problema. Este un método alternativo que inicia usando el conjunto de testores típicos, descartando características irrelevantes o redundantes.

Los 20 atributos descartados hacían referencia al error estándar y valores mayores de cada uno de los 10 atributos reales calculados para cada núcleo celular, siendo descartados en el presente ejercicio para facilitar el entendimiento del modelo de clasificación elaborado y no afectar su precisión.

Se definen entonces:

- Número de caracteristicas: $n_x$

- Número de muestras/registros totales: $m$

$$ X \in \mathbb{R}^{n_x\times m} \rightarrow X \in \mathbb{R}^{(10\times 569)} $$

$$ Y \in \mathbb{N}^{n_x \times m} \rightarrow Y \in \mathbb{N}^{(1 \times 569)}, \hspace{5pt} Y = \{0, 1\} $$

### Partición de los Datos

Siguiendo las recomendaciones para el entrenamiento de algoritmos ML, decidimos dividir o partir los datos totales ($m=569$) en $80%$ para entrenamiento, es decir, para ajustar los pesos, y 20% para evaluar la red neuronal. 

$$
X_{train} = 455, \hspace{10pt} con \hspace{3pt} forma \hspace{3pt} (10 \times 455)
$$

$$
X_{test} = 114, \hspace{10pt} con \hspace{3pt} forma \hspace{3pt} (10 \times 114)
$$

## Tipo de Tarea

Clasificación Binaria (dados los registros clinicos de un paciente sobre una masa tumoral predecir si es maligna o benigna).

- $0$ $\rightarrow$ Tumor Benigno
- $1$ $\rightarrow$ Tumor Maligno

## Capas de la Red Neuronal Artificial

### Notación

- Número de neuronas: $n$
- Capas Totales: $L$
- Capa individual: $l$
- Pesos: $W$
- Bias: $b$
- Preactivación: $Z$
- Activación: $A$

### Capa de Entrada
La capa de entrada de la red neuronal recibirá los datos, elegimos una capa con 10 perceptrones o neuronas ($n=10$), de modo que cada neurona reciba una caracteristica de la base de datos.

Cabe mencionar que la capa de entrada realmente no se cuenta cuando se hablan de capas totales en una arquitectura de red neuronal (solo se cuentan las capas ocultas y la capa de salida).

<img src="https://github.com/Ricardo-OB/redes_neuronales/blob/main/images/capa_entrada_2.png?raw=true" width="400">

<center>Figura 6. Representación capa de entrada </center>

La matriz de etiquetas a predecir (tambien denominadas $labels$) tienen la siguiente forma:

$$ Y = [y_0, y_1, y_2, y_3, y_4, ... , y_{568}] $$

### Capas Ocultas
Decidimos implementar 4 capas ocultas ($L_{ocultas}=4$) totalmente conectadas. Cada capa oculta en la Figura 7 se denota como $a_{i}^{[l]}$, donde $i$ denota el número de la neurona y $l$ el número de la capa. La configuración de neuronas se describe a continuación:

- Primera capa oculta ($l=1$) con 10 neuronas ($n=10$).
- Segunda capa oculta ($l=2$) con 10 neuronas ($n=12$).
- Tercera capa oculta ($l=3$) con 10 neuronas ($n=9$).
- Cuarta capa oculta ($l=4$) con 10 neuronas ($n=5$).

<img src="https://github.com/Ricardo-OB/redes_neuronales/blob/main/images/capas_ocultas.png?raw=true" width="440">

<center>Figura 7. Capas ocultas</center>

- La **primera capa oculta** tendrá una matriz de pesos $W^{[1]}$ de tamaño ($10\times 10$) y una matriz de $sesgo$ $b$ de tamaño ($10\times 1$).
- La **segunda capa oculta** tendrá una matriz de pesos $W^{[2]}$ de tamaño ($12\times 10$) y una matriz de $sesgo$ $b$ de tamaño ($12\times 1$).
- La **tercera capa oculta** tendrá una matriz de pesos $W^{[3]}$ de tamaño ($9\times 12$) y una matriz de $sesgo$ $b$ de tamaño ($9\times 1$).
- La **cuarta capa oculta** tendrá una matriz de pesos $W^{[4]}$ de tamaño ($5\times 9$) y una matriz de $sesgo$ $b$ de tamaño ($5\times 1$).

**Nota:** Cada capa oculta también tendrá matrices asociadas $dW$ y $db$ (matrices con resultados de derivadas parciales de la función de perdida) del mismo tamaño que $W$ y $b$.

### Capa de Salida

Finalmente, la capa de salida ($l=5$) tiene dos neuronas ($n=2$) correspondientes a las etiquetas a predecir (tumor benigno y tumor maligno).

<img src="https://github.com/Ricardo-OB/redes_neuronales/blob/main/images/capa_salida.png?raw=true" width="300">

<center>Figura 8. Representación de capa de salida</center>

- La **capa de salida** tendrá una matriz de pesos $W^{[5]}$ de tamaño ($2\times 5$) y una matriz de $sesgo$ $b$ de tamaño ($2\times 1$).

**Nota:** Esta capa oculta también tendrá matrices asociadas $dW$ y $db$.

### Resumen de capas y matrices

Tabla con dimensiones de matrices por capa y representación de matrices.

|Capa   |Tamaño de $W$      |Tamaño de $b$     |Notación Pre-Activación                          |Tamaño de Pre-Activación|
|-------|-------------------|------------------|-------------------------------------------------|------------------------|
|Capa 1 | $$(10 \times 10)$$|$$(10 \times 1)$$ | $$Z^{[1]} = W^{[1]} \cdot X + b^{[1]}$$         |$$(10 \times 455)$$     |
|Capa 2 | $$(12 \times 10)$$|$$(12 \times 1)$$ | $$Z^{[2]} = W^{[2]} \cdot A^{[1]} + b^{[2]}$$   |$$(12 \times 455)$$     |
|Capa 3 | $$(9 \times 12)$$ |$$(9 \times 1)$$  | $$Z^{[3]} = W^{[3]} \cdot A^{[2]} + b^{[3]}$$   |$$(9 \times 455)$$      |
|Capa 4 | $$(5 \times 9)$$  |$$(5 \times 1)$$  | $$Z^{[4]} = W^{[4]} \cdot A^{[3]} + b^{[4]}$$   |$$(5 \times 455)$$      |
|Capa 5 | $$(2 \times 5)$$  |$$(2 \times 1)$$  | $$Z^{[5]} = W^{[5]} \cdot A^{[4]} + b^{[5]}$$   |$$(2 \times 455)$$      |

Recordemos que $W \cdot X + b$ equivale a las siguientes operaciones (en Python la suma de $b$ se realiza con 'broadcasting', por lo tanto la dimensión $1$ se transforma): 

$$ 
W = \begin{bmatrix}
w_{00}  & w_{01} & w_{02} \\
w_{10}  & w_{11} & w_{12} \\
w_{20}  & w_{21} & w_{22} 
\end{bmatrix}\;\;\; 
X = \begin{bmatrix}
x_{00}  & x_{01} & x_{02} \\
x_{10}  & x_{11} & x_{12} \\
x_{20}  & x_{21} & x_{22} 
\end{bmatrix} \;\;\; 
b =\begin{bmatrix}
b_0  \\
b_1  \\
b_2
\end{bmatrix}
$$

$$ 
WX + b = \begin{bmatrix}
(w_{00}x_{00} + w_{01}x_{10} + w_{02}x_{20}) + b_0 & (w_{00}x_{01} + w_{01}x_{11} + w_{02}x_{21}) + b_0 & \cdots \\
(w_{10}x_{00} + w_{11}x_{10} + w_{12}x_{20}) + b_1 & (w_{10}x_{01} + w_{11}x_{11} + w_{12}x_{21}) + b_1 & \cdots \\
(w_{20}x_{00} + w_{21}x_{10} + w_{22}x_{20}) + b_2 &  (w_{20}x_{01} + w_{21}x_{11} + w_{22}x_{21}) + b_2 & \cdots
\end{bmatrix}
$$ 

## Funciones de Activación

Las funciones de activación elegidas para las capas ocultas son la Tangente Hiperbólica (y cómo alternativa la función Sigmoide), y para la capa de salida una función Softmax.

#### Tangente Hiperbólica
La función tangente hiperbólica transforma los valores introducidos a una escala $[-1,1]$, donde los valores altos tienen de manera asintótica a $1$ y los valores muy bajos tienden de manera asintótica a $-1$ [[3]](https://www.diegocalvo.es/funcion-de-activacion-redes-neuronales/).

La tangente hiperbólica de un número real $x$ se designa mediante $tanh(x)$ y se define como el cociente entre el seno hiperbólico y el coseno hiperbólico del número real $x$ [[4]](https://es.wikipedia.org/wiki/Tangente_hiperb%C3%B3lica). Si se sustituye de acuerdo con las definiciones de seno hiperbólico y coseno hiperbólico, se obtiene una fórmula más directa para la tangente hiperbólica, a saber:

<table>
<tr>
<td>
$$tanh(x) = \dfrac{e^x - e^{-x}}{e^x + e^{-x}} = \dfrac{sinh(x)}{cosh(x)}$$
</td>
<td>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hyperbolic_Tangent.svg/1920px-Hyperbolic_Tangent.png" width="500">
</td>
</tr>
</table>

<center>Figura 9. Función Tangente Hiperbólica y gráfica</center>

### Sigmoide

La función sigmoide transforma los valores introducidos a una escala $[0,1]$, donde los valores altos tienen de manera asintótica a $1$ y los valores muy bajos tienden de manera asintótica a $0$ [[3]](https://www.diegocalvo.es/funcion-de-activacion-redes-neuronales/).

Su gráfica tiene una típica forma de "S", y a menudo la función sigmoide se refiere al caso particular de la función logística, cuya gráfica se muestra en la Figura 10 y que viene definida por la siguiente fórmula [[5]](https://es.wikipedia.org/wiki/Funci%C3%B3n_sigmoide):

<table>
<tr>
<td>
$$\sigma(x) = \dfrac{1}{1 - e^{-x}}$$
</td>
<td>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Funci%C3%B3n_sigmoide_01.svg/800px-Funci%C3%B3n_sigmoide_01.png" width="400">
</td>
</tr>
</table>

<center>Figura 10. Función Sigmoide y gráfica</center>

### Softmax

La función softmax, también conocida como **softargmax** o **función exponencial normalizada**, convierte un vector de $K$ números reales ($\vec{z}$) en una distribución de probabilidad de $K$ resultados posibles $(\sigma(\vec{z}))$, es decir, un vector de valores reales en el rango $[0, 1]$. Es una generalización de la función logística a múltiples dimensiones y se utiliza en la regresión logística multinomial [[6]](https://es.wikipedia.org/wiki/Funci%C3%B3n_SoftMax)[[7]](https://en.wikipedia.org/wiki/Softmax_function). 

La función softmax se usa para normalizar la salida de una red a una distribución de probabilidad sobre las clases de salida predichas. La función está dada por:

$$\sigma(\vec{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \hspace{9pt} para \hspace{3pt} i = 1, ..., K$$

## Función de Perdida o Coste

La función de coste elegida es Binary Cross Entropy (entropía cruzada binaria, en español). Binary Cross Entropy (también conocida como **pérdida logarítmica (logarithmic loss o log loss))** es una métrica modelo que rastrea el etiquetado incorrecto de la clase de datos por parte de un modelo, lo que penaliza el modelo si se producen desviaciones en la probabilidad de clasificar las etiquetas. Los valores bajos de pérdida logaritmica equivalen a valores de alta precisión [[8]](https://arize.com/blog-course/binary-cross-entropy-log-loss/).

$$ J = -\frac{1}{m} \sum_{i=1}^{m} \color{Red}{y_i \cdot log(\hat{y_i})} + \color{Blue}{(1-y_i) \cdot log(1-\hat{y_i}))} $$

- Donde $\hat{y_i}$ son las probabilidades predichas por el modelo para todos los $m$ datos. De manera similar puede verse como $a^{[L]}$, es decir, las probabilidades calculadas en la capa $L$.

- Y $y_i$ son las etiquetas verdaderas o $labels$, con valores $0$ o $1$.

### Gradiente Descendente

El Gradiente Descendente o descenso de gradiente es el algoritmo de aprendizaje automático iterativo más utilizado y popular. También es la base para otros algoritmos de optimización. El descenso de gradiente tiene la siguiente regla de actualización para el parámetro de peso ($w$) y bias ($b$):

$$ w(\tau) = w(\tau-1)-\eta \hspace{4pt} \triangledown J(w) $$

$$ b(\tau) = b(\tau-1)-\eta \hspace{4pt} \triangledown J(b) $$

Donde J(w) le hemos definido como la función de coste *Binary Cross Entropy* y $\eta$ es la taza de aprendizaje. Los gradientes están definidos asi:

$$ \triangledown w = \dfrac{\partial J}{\partial w} $$

$$ \triangledown b = \dfrac{\partial J}{\partial b} $$

## Métricas de Desempeño

### Matriz de Confusión

La matriz de confusión es una herramienta que permite la visualización del desempeño de un algoritmo que se emplea en aprendizaje supervisado. Cada columna de la matriz representa el número de predicciones de cada clase, mientras que cada fila representa a las instancias en la clase real. Uno de los beneficios de las matrices de confusión es que facilitan ver si el sistema está confundiendo dos clases [[9]](https://es.wikipedia.org/wiki/Matriz_de_confusi%C3%B3n).

<img src="https://github.com/Ricardo-OB/redes_neuronales/blob/main/images/matriz_conf.png?raw=true" width="800">

<center>Figura 11. Representación Matriz de Confusión</center>

### Accuracy

El Accuracy (precisión en español) es una métrica para evaluar los modelos de clasificación. Informalmente, la precisión es la fracción de predicciones que nuestro modelo acertó [[10]](https://developers.google.com/machine-learning/crash-course/classification/accuracy). Su formula para clasificación binaria se describe asi:

$$ Accuracy = \dfrac{TP + TN}{TP + TN + FP + FN} $$

### Recall

La métrica Recall se puede explicar diciendo, de todas las clases positivas, cuántas predijimos correctamente. También podemos pensarlo como ¿Qué proporción de positivos reales se identificó correctamente? [[11]](https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62) Su formula se define de la siguiente manera:

$$ Recall= \dfrac{TP}{TP + FN} $$

### F1-Score

El F1-Score o medida F nos ayuda a medir Recall y Precision al mismo tiempo. Utiliza la media armónica en lugar de la media aritmética castigando más los valores extremos. Al ser la media armónica de Precision y Recall representa simétricamente tanto la Precision como el Recall en una métrica [[12]](https://en.wikipedia.org/wiki/F-score):

$$ F1= \dfrac{2 \cdot Recall \cdot Precision}{Recall \cdot Precision} $$

## Esquema general


