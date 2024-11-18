# Proyecto1
Algoritmo PageRank

## UNAM Facultad de Ciencias
### Programación
### Grupo 9292
Profesora: María Sánchez

### Instalación:
Para comenzar, clona el repositorio del proyecto:
```
git clone https://github.com/RodrigoSHS/Proyecto1.git
```

### Librerías necesarias
Para ejecutar este proyecto, necesitas instalar las siguientes librerías:

#### Mac o Linux
```
pip3 install numpy matplotlib networkx pandas
```

#### Windows
```
pip install numpy matplotlib networkx pandas
```

### Resultados del Código
Este proyecto implementa el algoritmo PageRank para calcular la importancia relativa de las páginas web en una red. Aquí te explicamos qué obtendrás al ejecutar el programa:

- **Visualización de la Red**: Se genera un gráfico que muestra cómo están conectadas las diferentes páginas web usando NetworkX y Matplotlib. Los nodos representan las páginas y las aristas los enlaces entre ellas. Esta visualización te permitirá ver de manera intuitiva cómo se estructura la red.

- **Valores de PageRank**: El programa calcula y muestra los valores de PageRank para cada página web. Estos valores indican qué tan importante es cada página en la red, basándose en la cantidad y calidad de los enlaces que recibe de otras páginas.

- **Top 3 Páginas por PageRank**: Finalmente, se muestran las tres páginas con los valores de PageRank más altos. Esto ayuda a identificar cuáles son las páginas más influyentes dentro de la red, es decir, aquellas que tienen una mayor relevancia en función de los enlaces.

Ejemplo de salida:

```
Valores de PageRank:
Página bloomberg.com: 0.0268
Página businessinsider.com: 0.0331
Página cnn.com: 0.0228
Página federalreserve.gov: 0.2000
Página forbes.com: 0.0207
Página foxnews.com: 0.0412
Página gazeta.ru: 0.0363
Página huffpost.com: 0.0275
Página iz.ru: 0.0363
Página kommersant.ru: 0.0359
Página lenta.ru: 0.0368
Página mashable.com: 0.0318
Página npr.org: 0.0286
Página nytimes.com: 0.0238
Página politico.com: 0.0250
Página rbk.ru: 0.0468
Página regnum.ru: 0.0362
Página reuters.com: 0.0256
Página ria.ru: 0.0365
Página tass.ru: 0.0365
Página techcrunch.com: 0.0191
Página theverge.com: 0.0200
Página vedomosti.ru: 0.0469
Página washingtonpost.com: 0.0308
Página wsj.com: 0.0386
Página yandex.ru: 0.0364

Top 3 páginas por PageRank:
Página federalreserve.gov: 0.2000
Página vedomosti.ru: 0.0469
Página rbk.ru: 0.0468
```
