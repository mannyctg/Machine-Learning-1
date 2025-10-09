"""
Librería de Funciones para graficos y calculos para el correcto analisis de datos.
================================

Una librería personalizada para operaciones de álgebra lineal con vectores y matrices.

Módulos disponibles:
- Graficos
- Calculos y analisis

"""
 
from .visual import (

    # Graficos

    multiple_plot,
    plot_roc_curve,
    tidy_corr_matrix,
    plot_dendrogram,

    # Calculos y analisis
    checkVIF,
    silhouette_analysis,
)

__version__ = "1.0.0"
__author__ = "Manuel Alejandro Zuñiga Navarro"
__all__ = [
    'multiple_plot',
    'plot_roc_curve',
    'tidy_corr_matrix',
    'plot_dendrogram',
    'checkVIF',
    'silhouette_analysis',
]