# Resumen Red Neuronal

## Definición
Las redes neuronales son modelos computacionales —inspirados en las neuronas que constituyen el cerebro (ver Imagen 1a y 1b)— y que dotan a los ordenadores de *Inteligencia* Artificial. Están formadas por unidades básicas llamadas neuronas o perceptrones (ver Imagen 3) que se conectan entre sí formando la red neuronal [[1]](https://enzyme.biz/blog/redes-neuronales-que-son-y-aplicaciones). El objetivo de la red neuronal es resolver los problemas de la misma manera que el cerebro humano, aunque las redes neuronales son más abstractas y su comparación sigue siendo muy distante. Al igual que las neuronas biológicas, los perceptrones están diseñados para tener dos salidas diferentes 1 o 0 (salida/no salida) [[2]](https://es.wikipedia.org/wiki/Red_neuronal_artificial).

Las redes neuronales pueden ayudar a las computadoras a tomar decisiones inteligentes con asistencia humana limitada. Esto se debe a que pueden *aprender* y modelar las relaciones entre los datos de entrada y salida [[3]](https://aws.amazon.com/es/what-is/neural-network/). Una red neuronal puede *aprender* o entrenarse a partir de datos de entrada mediante un ajuste de los pesos de las conexiones entre ellas.

![<img src="https://ml4a.github.io/images/neuron-anatomy.jpg" width="400" alt="Imagen 1">](https://ml4a.github.io/images/neuron-anatomy.jpg)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![<img src="https://ml4a.github.io/images/neuron-simple.jpg" width="300" alt="Imagen 1b">](https://ml4a.github.io/images/neuron-simple.jpg)

Imagen 1a. Representación de una neurona biológica [[4]](https://ml4a.github.io/images/neuron-anatomy.jpg). &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Imagen 1b. Simplificación de una neurona biológica [[4]](https://ml4a.github.io/images/neuron-simple.jpg).


## Pipeline


### Componentes
Una red neuronal consta de una capa de entrada, una de salida, y en el intermedio una cantidad $n$ de capas ocultas (ver Imagen 2). Cada capa está conformada por perceptrones o neuronas artificiales, y cada conexión entre dos neuronas tiene un peso asociado $W$, el cual representa la importancia relativa de la entrada que recibe la neurona de la conexión en la salida que produce [[18]](https://ml4a.github.io/ml4a/es/neural_networks/).

<img src="https://s7280.pcdn.co/wp-content/uploads/2020/07/Two-or-more-hidden-layers-comprise-a-Deep-Neural-Network.png" width="500" alt="Imagen 2">

Imagen 2. Estructura de una red neuronal artificial [[5]](https://s7280.pcdn.co/wp-content/uploads/2020/07/Two-or-more-hidden-layers-comprise-a-Deep-Neural-Network.png).

**Capa de Entrada:**
Recibe o contiene la información del mundo exterior (p. ej. imágenes, señales, registros). Comúnmente esta capa no hace parte del número de capas de la red [[19]](https://aws.amazon.com/es/what-is/neural-network/).

**Capa Oculta:**
Las capas ocultas toman su entrada de la capa de entrada o de otras capas ocultas. Cada capa oculta analiza la salida de la capa anterior, la procesa aún más y la pasa a la siguiente capa [[19]](https://aws.amazon.com/es/what-is/neural-network/).

**Capa de Salida:**
Proporciona el resultado final de todo el procesamiento de datos. Puede tener una o varias neuronas. Por ejemplo, en un problema de clasificación binaria (0/1), la capa de salida tendrá una neurona de salida que dará como resultado 1 o 0. Sin embargo, si es un problema de clasificación multiclase, la capa de salida puede tener más de una neurona de salida [[19]](https://aws.amazon.com/es/what-is/neural-network/).

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*v88ySSMr7JLaIBjwr4chTw.jpeg" width="550" alt="Imagen 3">

Imagen 3. Estructura del perceptrón o neurona artificial [[6]](https://miro.medium.com/v2/resize:fit:720/format:webp/1*v88ySSMr7JLaIBjwr4chTw.jpeg).


### Entrenamiento y Funcionamiento
El proceso de entrenamiento de una red neuronal se divide en dos fases: Forward-propagation (propagación hacia adelante) y Back-propagation (propagación hacia atrás). Cada una con diferentes pasos, los cuales involucran variados componenentes matemáticos.

#### Forward Propagation

#### Back Propagation

## Matemáticas utilizadas
1. Algebra Lineal: operaciones entre matrices como producto punto, suma, etc...
	* forward y back
2. Transformaciones no Lineales: funciones de activación como ReLU o Sigmoide, etc...
	* forward
4. Derivadas Parciales: para calcular el gradiente de la función de pérdida...
	* back
5. Regla de la cadena: para calcular el gradiente de una función compuesta de varias capas...
	* back
6. Optimización: algoritmos de optimización como el descenso de gradiente estocástico (SGD)
	* back
7. Convoluciones: en redes neuronales convolucionales
	* forward

## Ejemplos de aplicaciones y usos
Las redes neuronales están presentes en nuestro día a día en muchos sectores, como los siguientes:
* **Salud:**
	* Diagnóstico médico mediante la clasificación de imágenes médicas [[7]](https://arxiv.org/abs/1711.05225).
	* Segmentación de órganos o tejidos en imágenes [[8]](https://kits21.kits-challenge.org/).
* **Marketing:**
	* Filtrado de redes sociales y el análisis de datos de comportamiento [[9]](https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/).
* **Finanzas:**
	* Predicciones financieras mediante el procesamiento de datos históricos [[10]](https://zenodo.org/record/2626454).
	* Detección de fraude financiero [[11]](https://www.sciencedirect.com/science/article/abs/pii/S0167923620301767).
* **Industria:**
	* Identificación de compuestos químicos [[12]](https://www.sciencedirect.com/science/article/pii/S2001037022003300).
	* Proceso y control de calidad [[13]](https://www.redalyc.org/journal/404/40465051005/movil/).
* **Agricultura:**
	* Mapeo de rendimiento de terrenos [[14]](https://www.sciencedirect.com/science/article/pii/S258972172030012X).
	* Detección de maleza [[15]](https://tensorfield.ag/2020/04/14/weed-detection-demo/).
* **Justicia y Seguridad:**
	* Reconocimiento facial [[16]](https://www.nytimes.com/2019/04/14/technology/china-surveillance-artificial-intelligence-racial-profiling.html).
	* Predicción de crímenes [[17]](https://land-der-ideen.de/en/project/precobs-software-for-predicting-crimes-355).
* **Otros:**
	* Motores de búsqueda o traducción.
	* Recomendación de productos.
	* Asistentes virtuales (Alexa, Siri, Google Assistant).

## Planteamiento del problema

*En el presente proyecto los autores diseñarán una red neuronal artificial para el diagnóstico de cáncer de mama (clasificación binaria).*

El cáncer de mama es el tumor maligno más frecuente en el mundo y su investigación constituye un desafío en la práctica médica. El diagnostico precoz del cáncer de mama se ve obstaculizado por la heterogeneidad clínica y patológica de la enfermedad y las limitaciones de los métodos de cribado utilizados hasta el momento [[20]](http://scielo.sld.cu/pdf/rcim/v13n1/1684-1859-rcim-13-01-e385.pdf). Según la Agencia Internacional para la Investigación del Cáncer (IARC), sección de vigilancia de cáncer de la Organización Mundial de la Salud, en el 2020 se diagnosticaron en el mundo un total de 19 292 789 casos de cáncer en ambos sexos para todas las edades. Los de mayor frecuencia según orden fueron el cáncer de mama con 2 261 419 casos (11.7%), seguido del cáncer de pulmón con 2 206 771 casos (11.4%) y luego el cáncer colorrectal con 1 931 590 casos (10%). El número total de muertes por cáncer ese mismo año fue de 9 958 133 casos para ambos sexos, a su vez el número de muertes por en el cáncer del pulmón fue de 1 796 144 casos (18%), seguido de 935 173 casos (9.4%) de fallecidos por cáncer colorrectal y 684 996 casos (6.9%) de los fallecidos cáncer de mama [[21]](https://gco.iarc.fr/today/data/factsheets/cancers/20-Breast-fact-sheet.pdf).

Su diagnóstico se basa en la obtención de los factores de riesgo, los hallazgos al examen físico, las pruebas de imágenes y exámenes anatomopatológicos, siendo de gran importancia el uso de datos clinicos de la masa, asi como Mamografías y TAC para realizar este diagnóstico. El control del cáncer de mama es un problema que amerita una solución rápida. Todo lo mencionado llevó a los autores a plantearse la creación de una red neuronal artificial con Python para la clasificación de tumores benignos o malignos en el seno mediante el usos de datos clinicos, y realizar un diagnóstico precoz del cáncer de mama.

