# Umgebung
Die Skripty basieren auf Python 3.8 mit Spyder als Entwicklungsumgebung.
Getestet mit [Anaconda Python Distribution](https://anaconda.org/) bzw. [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
Es gibt noch [alternative Distributionen](https://wiki.python.org/moin/PythonDistributions) wie [Python(x,y)](https://python-xy.github.io/).

# Packages
List der benutzten Packages (noch zu ergänzen):
* spyder
* numpy
* scipy
* pandas
* matplotlib
* seaborn
* pyarrow
* fastparquet
* ...

# Python Skripte
* fig_util.py: 

# Anaconda/Miniconda
Die folgenden Infos sind für die Miniconda/Anaconda Distribution.

## Update: Anaconda Prompt (Kommandozeile, als Administrator ausführen)
```
conda update conda
```

## Erstellen einer eigenen Umgebung
```
conda create -n Cov19 python=3.8
```


## Wechseln in die eigene Umgebung
```
conda activate Cov19
```

## Update/Installation der Anaconda Packages
```
conda install anaconda
conda update anaconda
conda install -c conda-forge pyarrow
```

## Spyder
```
spyder
```
Über Menü>Projects>Open Projekt den "src" Ordner auswählen.
