import sys
print("Version Python :\n", sys.version)

from pip import _internal
print("PIP LIST :\n", _internal.main(['list']))

from pyspark import SparkContext
sc = SparkContext()   # inutile mais nécessaire pour que l'EMR accepte le job
