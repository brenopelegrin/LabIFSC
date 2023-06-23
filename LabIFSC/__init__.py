#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Gabriel Queiroz"
__credits__ = ["gabriel Queiroz", "Pedro Ilídio"]
__license__ = "MIT"
__version__ = "0.1.13"
__email__ = "gabrieljvnq@gmail.com"
__status__ = "Production"

from .geral import TODAS_AS_UNIDADES, MAPA_DE_DIMENSOES, PREFIXOS_SI_LONGOS, PREFIXOS_SI_CURTOS, PREFIXOS_SI, analisa_numero, analisa_unidades, calcula_dimensao, parse_dimensions, acha_unidade, unidades_em_texto
from .medida import Medida, M, arrayM, MCarlo, montecarlo, LabIFSC_Mcarlo_samples
from .unidade import Unidade
from .lista_de_unidades import registra_unidades ; registra_unidades()
from .matematica import soma, cos, sin, tan, arc_cos, arc_sin, arc_tan, log, log10, log2, ln, sqrt, cbrt, dam, mean
from .tabela import media, desvio_padrao, linearize, compare, Tabela
from .constantes import *


__all__ = [
    "TODAS_AS_UNIDADES", "MAPA_DE_DIMENSOES", "PREFIXOS_SI_LONGOS", "PREFIXOS_SI_CURTOS", "PREFIXOS_SI", "analisa_numero", "analisa_unidades", "calcula_dimensao", "parse_dimensions", "acha_unidade", "unidades_em_texto",
    "Medida", "M","MCarlo", "arrayM", "Tabela",
    "Unidade",
    "registra_unidades",
    "soma", "cos", "sin", "tan", "arc_cos", "arc_sin", "arc_tan", "log", "log10", "log2", "ln", "sqrt", "cbrt",
    "media", "desvio_padrao", "linearize", "compare","montecarlo"] + nomes_constantes
