
# Desarrollo Teórico 

**Integrantes**
- Ricardo Ortega Bolaños.
- Viowi Y. Cabrisas Amuedo.

## Base de Datos y Problema

El problema a resolver trata sobre el cáncer de mama, uno de los tumores malignos más frecuentes en el mundo y su investigación constituye un desafío en la práctica médica. El diagnóstico precoz del cáncer de mama se ve obstaculizado por la heterogeneidad clínica y patológica de la enfermedad y las limitaciones de los métodos de cribado utilizados hasta el momento [[1]](http://scielo.sld.cu/pdf/rcim/v13n1/1684-1859-rcim-13-01-e385.pdf).

El diagnóstico, entre tumor benigno y maligno, se basa en la obtención de los factores de riesgo, los hallazgos al examen físico, las pruebas de imágenes y exámenes anatomopatológicos, siendo de gran importancia el uso de datos clínicos de la masa, así como Mamografías y TAC.

En nuestro caso, nos enfocamos en el diagnostico precoz de cáncer de mama a partir de datos clínicos. La base de datos elegida [[2]](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) pertenece a la Universidad de Wisconsin, contiene …


- Número de caracteristicas: $n_x$

- Número de muestras/registros: $m$

$$ X \in \mathbb{R}^{n_x\times m} \rightarrow X \in \mathbb{R}^{(10\times 569)} $$

$$ Y \in \mathbb{N}^{n_x \times m} \rightarrow Y \in \mathbb{N}^{(1 \times 569)}, \,\,\,\, Y = \{0, 1\} $$

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

<img src="https://github.com/Ricardo-OB/redes_neuronales/blob/main/images/capa_entrada.png?raw=true" width="400">

<center>Imagen 1. Representación capa de entrada </center>

La matriz de etiquetas a predecir (tambien denominadas $labels$) tienen la siguiente forma:

$$ Y = [y_0, y_1, y_2, y_3, y_4, ... , y_{568}] $$

### Capas Ocultas
Decidimos implementar 4 capas ocultas ($L_{ocultas}=4$) totalmente conectadas. Cada capa oculta en la Imagen 2 se denota como $a_{i}^{[l]}$, donde $i$ denota el número de la neurona y $l$ el número de la capa. La configuración de neuronas se describe a continuación:

- Primera capa oculta ($l=1$) con 10 neuronas ($n=10$).
- Segunda capa oculta ($l=2$) con 10 neuronas ($n=12$).
- Tercera capa oculta ($l=3$) con 10 neuronas ($n=9$).
- Cuarta capa oculta ($l=4$) con 10 neuronas ($n=5$).

<img src="https://github.com/Ricardo-OB/redes_neuronales/blob/main/images/capas_ocultas.png?raw=true" width="440">

<center>Imagen 2. Capas ocultas</center>

- La **primera capa oculta** tendrá una matriz de pesos $W^{[1]}$ de tamaño ($10\times 10$) y una matriz de $sesgo$ $b$ de tamaño ($10\times 1$).
- La **segunda capa oculta** tendrá una matriz de pesos $W^{[2]}$ de tamaño ($12\times 10$) y una matriz de $sesgo$ $b$ de tamaño ($12\times 1$).
- La **tercera capa oculta** tendrá una matriz de pesos $W^{[3]}$ de tamaño ($9\times 12$) y una matriz de $sesgo$ $b$ de tamaño ($9\times 1$).
- La **cuarta capa oculta** tendrá una matriz de pesos $W^{[4]}$ de tamaño ($5\times 9$) y una matriz de $sesgo$ $b$ de tamaño ($5\times 1$).

**Nota:** Cada capa oculta también tendrá matrices asociadas $dW$ y $db$ (matrices con resultados de derivadas parciales de la función de perdida) del mismo tamaño que $W$ y $b$.

### Capa de Salida

Finalmente, la capa de salida ($l=5$) tiene dos neuronas ($n=2$) correspondientes a las etiquetas a predecir (tumor benigno y tumor maligno).

<img src="https://github.com/Ricardo-OB/redes_neuronales/blob/main/images/capa_salida.png?raw=true" width="300">

<center>Imagen 3. Representación de capa de salida</center>

- La **capa de salida** tendrá una matriz de pesos $W^{[5]}$ de tamaño ($2\times 5$) y una matriz de $sesgo$ $b$ de tamaño ($2\times 1$).

**Nota:** Esta capa oculta también tendrá matrices asociadas $dW$ y $db$.

### Resumen de capas y matrices

Tabla con dimensiones de matrices por capa y formulas.

## Funciones de Activación

Las funciones de activación elegidas para las capas ocultas son la Tangente Hiperbólica (y cómo alternativa la función Sigmoide), y para la capa de salida una función Softmax.

#### Tangente Hiperbólica
La función tangente hiperbólica transforma los valores introducidos a una escala $[-1,1]$, donde los valores altos tienen de manera asintótica a $1$ y los valores muy bajos tienden de manera asintótica a $-1$ [[3]](https://www.diegocalvo.es/funcion-de-activacion-redes-neuronales/).

La tangente hiperbólica de un número real $x$ se designa mediante $tanh(x)$ y se define como el cociente entre el seno hiperbólico y el coseno hiperbólico del número real $x$ [[4]](https://es.wikipedia.org/wiki/Tangente_hiperb%C3%B3lica). Si se sustituye de acuerdo con las definiciones de seno hiperbólico y coseno hiperbólico, se obtiene una fórmula más directa para la tangente hiperbólica, a saber:

<table>
  <tr>
    <td>
        $tanh(x) = \dfrac{e^x - e^{-x}}{e^x + e^{-x}} = \dfrac{sinh(x)}{cosh(x)}$
    </td>
    <td>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Hyperbolic_Tangent.svg/1920px-Hyperbolic_Tangent.png" width="500">
    </td>
  </tr>
</table>

<center>Imagen 4. Función Tangente Hiperbólica y gráfica</center>

### Sigmoide

La función sigmoide transforma los valores introducidos a una escala $[0,1]$, donde los valores altos tienen de manera asintótica a $1$ y los valores muy bajos tienden de manera asintótica a $0$ [[3]](https://www.diegocalvo.es/funcion-de-activacion-redes-neuronales/).

Su gráfica tiene una típica forma de "S", y a menudo la función sigmoide se refiere al caso particular de la función logística, cuya gráfica se muestra en la Imagen 5 y que viene definida por la siguiente fórmula [[5]](https://es.wikipedia.org/wiki/Funci%C3%B3n_sigmoide):

<table>
  <tr>
    <td>
        $\sigma(x) = \dfrac{1}{1 - e^{-x}}$
    </td>
    <td>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Funci%C3%B3n_sigmoide_01.svg/800px-Funci%C3%B3n_sigmoide_01.png" width="400">
    </td>
  </tr>
</table>

<center>Imagen 5. Función Sigmoide y gráfica</center>

### Softmax

La función softmax, también conocida como **softargmax** o **función exponencial normalizada**, convierte un vector de $K$ números reales ($\vec{z}$) en una distribución de probabilidad de $K$ resultados posibles ($\sigma(\vec{z})$), es decir, un vector de valores reales en el rango $[0, 1]$. Es una generalización de la función logística a múltiples dimensiones y se utiliza en la regresión logística multinomial [[6]](https://es.wikipedia.org/wiki/Funci%C3%B3n_SoftMax)[[7]](https://en.wikipedia.org/wiki/Softmax_function). 

La función softmax se usa para normalizar la salida de una red a una distribución de probabilidad sobre las clases de salida predichas. La función está dada por:

$$ \sigma(\vec{z})_i = \dfrac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

## Función de Perdida o Coste

Binary Cross Entropy

Gradiente (Jacobiano/Hessiano) de Binary Cross Entropy

## Métricas de Desempeño

Accuracy

Recall

F1-Score

Matriz de Confusión

Curva ROC

## Esquema general
