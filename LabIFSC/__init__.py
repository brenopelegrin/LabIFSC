#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Gabriel Queiroz"
__credits__ = ["gabriel Queiroz", "Pedro Ilídio"]
__license__ = "MIT"
__version__ = "0.1.13"
__email__ = "gabrieljvnq@gmail.com"
__status__ = "Production"

from .geral import TODAS_AS_UNIDADES, MAPA_DE_DIMENSOES, PREFIXOS_SI_LONGOS, PREFIXOS_SI_CURTOS, PREFIXOS_SI, analisa_numero, analisa_unidades, calcula_dimensao, parse_dimensions, acha_unidade, unidades_em_texto
from .medida import Medida, M,MCarlo,montecarlo
from .unidade import Unidade
from .lista_de_unidades import registra_unidades
from .matematica import soma, cos, sin, tan, arc_cos, arc_sin, arc_tan, log, log10, log2, ln, sqrt, cbrt, dam, mean
from .tabela import media, desvio_padrao, linearize, compare

__all__ = [
    "TODAS_AS_UNIDADES", "MAPA_DE_DIMENSOES", "PREFIXOS_SI_LONGOS", "PREFIXOS_SI_CURTOS", "PREFIXOS_SI", "analisa_numero", "analisa_unidades", "calcula_dimensao", "parse_dimensions", "acha_unidade", "unidades_em_texto",
    "Medida", "M","MCarlo",
    "Unidade",
    "registra_unidades",
    "soma", "cos", "sin", "tan", "arc_cos", "arc_sin", "arc_tan", "log", "log10", "log2", "ln", "sqrt", "cbrt",
    "media", "desvio_padrao", "linearize", "compare","montecarlo"
]

def init():
    global TODAS_AS_UNIDADES
    registra_unidades()

init()
