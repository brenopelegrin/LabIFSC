#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import sys
from copy import copy
from .geral import TODAS_AS_UNIDADES, acha_unidade, calcula_dimensao, analisa_numero, dimensao_em_texto, fator_de_conversao_para_si, unidades_em_texto, converte_unidades, analisa_unidades, simplifica_unidades, gera_expoente, adimensional, get_unidades

import sys
PY3 = sys.version_info[0] == 3
if PY3:
    string_types = str
else:
    string_types = basestring


LabIFSC_Mcarlo_samples=1e4

def montecarlo(func, *args, comparar=False, N=LabIFSC_Mcarlo_samples, hist=False, bins=100, probabilidade=False):
    '''Propagação de erros usando monte carlo

  Calcula a densidade de probabilidade do valor de uma função com variaveis gaussianas,
  é possível visualizar o histograma com a biblioteca matplotlib instalada e acelerar
  o código com numpy, é possivel calcular a probabilidade

  ela retorna uma Medida LabIFSC com a media e desvio padrão da distribuição

  Args:
    func (function) : função para a propagação do erro
    args (Medidas LabIFSC) : parametros da função definada acima
    comparar (bool) : ativa ou não a comparação do calculo com uma aproximação linear da biblioteca do LabIFSC
    N (int) : quantidade de numeros aleatorios usados
    hist (bool) : ativar ou não a visualização do histograma (necessário matplotlib instalado)
    bins (int) : quantidade de bins usadas no histograma
    probabilidade(list) : uma lista [a,b] em que a é o começo do intervalo e b o fim

  Returns:
     MCarlo(media,desviopadrao,unidade)
  Raises:
    func e/ou args não existem ValueError("Não foi especificado uma função ou seus parametros")
    um dos args não é uma medida ValueError("Todos os parametros precisam ser Medida ou MCarlo")
    N e/ou bins não são inteiros ValueError(""N e bins precisam ser inteiros")
    comparar e/ou hist não são booleanos ValueError("Comparar e hist precisam ser booleanos")
'''
    # verificando se matplot e numpy estam instalados
    from sys import modules
    instalados = []
    if "matplotlib.pyplot" in modules: instalados.append("matplot")
    if "numpy" in modules: instalados.append("numpy")
    # importando variaveis e mensagens de erro
    try:
        funcao = func; parametros = args
    except:
        raise ValueError("Não foi especificado uma função ou seus parametros")
    for j in parametros:
        if not isinstance(j, Medida): 
            try: Medida((j.nominal,j.incerteza),j.unidade())
            except:raise ValueError("Todos os parametros precisam ser medidas ou MCarlo")
    try:
        N=int(N) ; bins=int(bins)
    except:
        raise ValueError("N e bins precisam ser inteiros")
    if not isinstance(comparar, bool) or not isinstance(hist, bool):
        raise ValueError("Comparar e hist precisam ser booleanos")
    if not isinstance(probabilidade, bool) and not isinstance(probabilidade, list):
        raise ValueError("Probabilidade é uma lista [a,b] em que a é o inicio e b o fim do intervalo")
    if type(probabilidade) == list:
        assert len(probabilidade) == 2;
        "Probabilidade é uma lista [a,b] em que a é o inicio e b o fim do intervalo"
    # criando numeros aleatorios gaussianos
    amostras = []
    if "numpy" in instalados:
        from numpy.random import normal
        for j in range(len(parametros)):  # numeros aleatorios com numpy
            amostras.append((normal(parametros[j].nominal, parametros[j].incerteza, N)))
    else:
        from random import gauss
        for j in range(len(parametros)):  # numeros aleatorios com random
            amostras.append([gauss(parametros[j].nominal, parametros[j].incerteza) for _ in range(N)])
    # calculando a função nos numeros aleatorios
    valores = []
    for k in range(N):
        parametros_funcao = [amostras[j][k] for j in range((len(parametros)))]
        try:
            valores.append(funcao(*parametros_funcao).nominal)  # caso estejamos usando uma função do LabIFSC
        except:
            valores.append(funcao(*parametros_funcao))  # funções não nativas do LabIFSC
    # media e desvio_padrao
    if "numpy" in instalados:  # média e desvio-padrão usando numpy
        import numpy as np
        media = np.average(valores)
        desviopadrao = np.std(valores)
    else:  # usando somente python
        media = sum(valores) / len(valores)
        desviopadrao = 0
        for j in valores: desviopadrao += (media - j) ** 2
        desviopadrao = (desviopadrao / len(valores)) ** (1 / 2)
    if probabilidade != False:
        a = probabilidade[0]
        b = probabilidade[1]
        counter = 0
        for j in valores:
            if a <= j <= b: counter += 1
        return counter/len(valores)
    # criando histograma
    if hist == True and "matplot" not in instalados:
        raise ValueError("Hist=true porém, você não possue o matplotlib instalado")
    if "matplot" in instalados and hist == True:
        import matplotlib.pyplot as plt
        plt.hist(valores, bins, density=True, label="Simulação")
        plt.xlabel('Valores')
        plt.ylabel('Densidade de probabilidade')
        plt.title(f"Monte Carlo N={N:,d}")
        if "numpy" in instalados:  # plotando melhor gaussiana usando numpy
            def gaussiana_nump(x, mu, sigma):
                norm = 1. / (sigma * (2. * 3.14159) ** 0.5)
                return norm * np.exp(-(x - mu) ** 2. / (2. * sigma ** 2))

            x = np.linspace(min(valores), max(valores), 1000)
            y = gaussiana_nump(x, media, desviopadrao)
        else:  # #plotando melhor gaussiana usando a bibllioteca math
            from math import exp
            def gaussiana(x, mu, sigma):
                norm = 1. / (sigma * (2. * 3.14159) ** 0.5)
                return norm * exp(-(x - mu) ** 2. / (2. * sigma ** 2))

            print(max(valores), min(valores))

            def linspace(comeco, fim, particoes):
                passos = (fim - comeco) / (particoes - 1)
                return [comeco + i * passos for i in range(particoes)]

            x = linspace(min(valores), max(valores), N)
            y = [gaussiana(i, media, desviopadrao) for i in x]
        plt.plot(x, y, label="Melhor Gaussiana")
        plt.legend()
        plt.show()
    try:  # tenta rodar a função recebendo uma medida como parametro
        linear = funcao(*parametros)
    except:
        linear = False
        if comparar == True: print("Função não é nativa do LabIFSC, logo a comparação não é possível")
    try:
        unidade = linear.unidade()  # tenta extrar a unidade
    except:
        unidade = ""
    if comparar == True and isinstance(linear, Medida): print(f"Esse é o resultado linear {linear}")
    return MCarlo((media, desviopadrao), unidade)



def M(*args, **kwargs):
    if len(args) > 0:
        if isinstance(args[0], list) or type(args[0]).__name__ == 'ndarray':
            if type(args[0]).__module__ == 'numpy' and type(args[0]).__name__ == 'ndarray':
                lista = args[0].tolist()
                try:
                    modulename = type(args[0]).__module__
                    numpy_module = sys.modules[modulename]
                except:
                    raise ModuleNotFoundError("a biblioteca numpy não foi encontrada, então o suporte à medidas em numpy.ndarray não irá funcionar")
                ans = []
                for m in lista:
                    if isinstance(m, Medida):
                        ans.append(m)
                    else:
                        ans.append(Medida(m, **kwargs))
                return numpy_module.array(ans)
            if isinstance(args[0], list):
                lista = args[0]
                ans = []
                for m in lista:
                    if isinstance(m, Medida):
                        ans.append(m)
                    else:
                        ans.append(Medida(m, **kwargs))
                return ans
    else:
        return Medida(*args, **kwargs)

class Medida:
    unidades_originais = [] # Tuplas (objeto unidade, expoente) na ordem em que foram entradas  
    dimensao = (0, 0, 0, 0, 0, 0, 0)
    nominal = 0.0
    incerteza = 0.0
    si_nominal = 0.0
    si_incerteza = 0.0

    def __init__(self, valor, unidade=None, incerteza=None):
        if isinstance(valor, list) or type(valor).__name__ == 'ndarray':
            raise ValueError("use M([...]) para gerar medidas a partir de listas de números")

        if incerteza != None:
            #raise ValueError("a incerteza não foi informada, então o valor 0.0 foi assumido")
            self.inicializa((valor, incerteza), unidade_txt=unidade)
        else:
            self.inicializa(valor, unidade_txt=unidade)

    def inicializa(self, valor, unidade_txt=None):
        # Analise o valor
        if isinstance(valor, Medida):
            self.nominal = valor.nominal
            self.incerteza = valor.incerteza
        elif isinstance(valor, str):
            self.nominal, self.incerteza = analisa_numero(valor)
        elif isinstance(valor, tuple) and len(valor) == 2:
            self.nominal, self.incerteza = float(valor[0]), abs(float(valor[1]))
        else:
            try:
                self.nominal = float(valor)
            except:
                raise Exception("não foi possível extrair o valor e a incerteza")

        # Veja as unidades
        if isinstance(unidade_txt, list):
            self.unidades_originais = copy(unidade_txt)
        elif unidade_txt != None:
            self.unidades_originais = analisa_unidades(unidade_txt)
        self.dimensao = calcula_dimensao(self.unidades_originais)

        # Converta para o SI
        mul_nom, mul_err, add_nom, add_err = fator_de_conversao_para_si(self.unidades_originais)
        self.si_nominal = self.nominal * mul_nom + add_nom
        self.si_incerteza = (self.nominal * mul_err + mul_nom * self.incerteza)  + add_err

    def SI(self):
        m = Medida(0)
        # Copie os dados
        m.dimensao = self.dimensao
        m.si_nominal = self.si_nominal
        m.si_incerteza = self.si_incerteza
        m.nominal = self.si_nominal
        m.incerteza = self.si_incerteza

        # Gere as unidades
        global TODAS_AS_UNIDADES
        m.unidades_originais = []

        # Casos especiais
        casos_especiais = ["joule", "newton", "pascal", "watt", "coulomb", "ohm", "farad"]
        for caso_especial in casos_especiais:
            u = TODAS_AS_UNIDADES[caso_especial]
            if self.dimensao == u.dimensao:
                m.unidades_originais.append(u)
                return m

        # Caso padrão
        if self.dimensao[0] != 0:
            m.unidades_originais.append(TODAS_AS_UNIDADES['metro'].nova_unidade_por_expoente(self.dimensao[0]))
        if self.dimensao[1] != 0:
            m.unidades_originais.append(TODAS_AS_UNIDADES['radiano'].nova_unidade_por_expoente(self.dimensao[1]))
        if self.dimensao[2] != 0:
            m.unidades_originais.append(TODAS_AS_UNIDADES['quilograma'].nova_unidade_por_expoente(self.dimensao[2]))
        if self.dimensao[3] != 0:
            m.unidades_originais.append(TODAS_AS_UNIDADES['segundo'].nova_unidade_por_expoente(self.dimensao[3]))
        if self.dimensao[4] != 0:
            m.unidades_originais.append(TODAS_AS_UNIDADES['kelvin'].nova_unidade_por_expoente(self.dimensao[4]))
        if self.dimensao[5] != 0:
            m.unidades_originais.append(TODAS_AS_UNIDADES['ampère'].nova_unidade_por_expoente(self.dimensao[5]))
        if self.dimensao[6] != 0:
            m.unidades_originais.append(TODAS_AS_UNIDADES['mol'].nova_unidade_por_expoente(self.dimensao[6]))

        return m

    def unidade(self, separador=" ", estilo=""):
        return unidades_em_texto(self.unidades_originais, separador, estilo)

    def checa_dim(self, outro):
        # Aplique a regra
        if self.dimensao != outro.dimensao:
            raise ValueError("dimensões físicas incompatíveis: {} vs {} ({} vs. {})".format(dimensao_em_texto(self.dimensao), dimensao_em_texto(outro.dimensao), self, outro))

    def _eh_medida(self, outro):
        if not isinstance(outro, Medida):
            raise TypeError("medidas só podem ser comparadas com outras medidas")

    def _torne_medida(self, outro, converter):
        m = outro
        if not isinstance(outro, Medida):
            m = Medida(outro)
        if converter:
            return m.converta(self.unidades_originais, ignore=True)
        else:
            return m

    def converta(self, unidades, ignore=False):
        if isinstance(unidades, str):
            unidades = analisa_unidades(unidades)
        m = Medida(1, unidades)
        if not ignore:
            self.checa_dim(m)
        nom, err = converte_unidades(self.nominal, self.incerteza, self.unidades_originais, unidades)
        return Medida((nom, err), unidades)

    def __str__(self):
        return "{}".format(self)

    def __repr__(self):
        return "<{}±{} {} = {}±{} {}>".format(self.nominal, self.incerteza, unidades_em_texto(self.unidades_originais), self.si_nominal, self.si_incerteza, dimensao_em_texto(self.dimensao))

    def __eq__(self, outro):
        self._eh_medida(outro)
        self.checa_dim(outro)
        return abs(self.si_nominal - outro.si_nominal) <= 2 * (self.si_incerteza + outro.si_incerteza)

    def __ne__(self, outro):
        self._eh_medida(outro)
        self.checa_dim(outro)
        return abs(self.si_nominal - outro.si_nominal) > 3 * (self.si_incerteza + outro.si_incerteza)

    def __add__(self, outro):
        unidades = get_unidades(outro)
        outro = self._torne_medida(outro, True)
        if not adimensional(self):
            unidades = self.unidades_originais
        m = Medida((self.nominal+outro.nominal, self.incerteza+outro.incerteza), unidades)
        return m

    def __sub__(self, outro):
        unidades = get_unidades(outro)
        outro = self._torne_medida(outro, True)
        if not adimensional(self):
            unidades = self.unidades_originais
        m = Medida((self.nominal-outro.nominal, self.incerteza+outro.incerteza), unidades)
        return m
    
    def __mul__(self, outro):
        outro = self._torne_medida(outro, False)
        nom = self.nominal * outro.nominal
        err = self.nominal * outro.incerteza + self.incerteza * outro.nominal
        m = Medida((nom, err), simplifica_unidades(self.unidades_originais, outro.unidades_originais))
        return m
    
    def __div__(self, outro):
        outro = self._torne_medida(outro, False)
        nom = self.nominal / outro.nominal
        err = (self.nominal * outro.incerteza + self.incerteza * outro.nominal)/outro.nominal**2
        m = Medida((nom, err), simplifica_unidades(self.unidades_originais, outro.unidades_originais, inverte=True))
        return m
    
    def __floordiv__(self, outro):
        m = self.__div__(outro)
        nom = math.floor(m.nominal)
        err = abs(m.nominal-nom) + m.incerteza
        return Medida((nom, err), m.unidades_originais)
    
    def __truediv__(self, outro):
        return self.__div__(outro)
    
    def __mod__(self, outro):
        pass
    
    def __divmod__(self, outro):
        outro = self._torne_medida(outro, False)
        a = self.__floordiv__(1)
        b = outro.__floordiv__(1)
        unidades = simplifica_unidades(self.unidades_originais, outro.unidades_originais, inverte=True)

        q_nom, r_nom = divmod(int(a.nominal), int(b.nominal))
        err = ((self.nominal * outro.incerteza) + self.incerteza * outro.nominal)/outro.nominal**2
        err *= 1/outro.nominal
        return Medida((q_nom, err), unidades), Medida((r_nom, err), unidades)
    
    def __pow__(self, outro):
        if not isinstance(outro, Medida):
            return self.__pow__(Medida(outro))
    
        if outro.dimensao != (0, 0, 0, 0, 0, 0, 0):
            raise NotImplementedError("o expoente deve ser adimensional")

        unidades = []
        if outro.nominal == int(outro.nominal) or (1/outro.nominal) == int(1/outro.nominal):
            for u in self.unidades_originais:
                unidades.append(u.nova_unidade_por_expoente(outro.nominal))
            unidades = simplifica_unidades(unidades)
        else:
            unidades = simplifica_unidades(self.unidades_originais)

        A = self.nominal
        B = outro.nominal
        # s = σ
        sA = self.incerteza
        sB = outro.incerteza
        sAB = sA*sB # Não tenho certeza se esse valor está certo

        f = self.nominal ** outro.nominal
        sf = 0
        sf += ((B/A)*sA)**2
        sf += (math.log(math.fabs(A))*sB)**2
        sf += 2*B*math.log(math.fabs(A))*sAB/A
        sf = math.fabs(f)*math.sqrt(math.fabs(sf))
        return Medida((f, sf), unidades)

    def __abs__(self):
        m = Medida((abs(self.nominal), self.incerteza), self.unidades_originais)
        return m
    
    def __int__(self):
        return int(self.nominal)
    
    def __float__(self):
        return float(self.nominal)
    
    def __complex__(self):
        return complex(self.nominal)
    
    def __radd__(self, outro):
        return Medida(outro).__add__(self)
    
    def __rsub__(self, outro):
        return Medida(outro).__sub__(self)
    
    def __rmul__(self, outro):
        return Medida(outro).__mul__(self)
    
    def __rfloordiv__(self, outro):
        return Medida(outro).__floordiv__(self)
    
    def __rmod__(self, outro):
        return Medida(outro).__mod__(self)
    
    def __rdivmod__(self, outro):
        return Medida(outro).__divmod__(self)
    
    def __rpow__(self, outro):
        outro = float(outro)
        return Medida((
            outro**self.nominal,
            outro**self.nominal * log(outro) * self.incerteza))
    
    def __rdiv__(self, outro):
        print(">", outro, self)
        return Medida(outro).__div__(self)

    def __rtruediv__(self, outro):
        return Medida(outro).__truediv__(self)

    def __format__(self, fmt):
        fmt = fmt.split(",")
        modo = fmt[0]
        exp = 0
        rouding = "ifsc"
        if len(fmt) >= 2:
            rouding = fmt[1]
        if len(fmt) >= 3:
            exp = int(fmt[2])
        expf = str(exp)
        self_nom = self.nominal*10**(-exp)
        self_err = self.incerteza*10**(-exp)
        nom = ""
        sep = ""
        err = ""
        uni = ""
        base = "({nom}±{err}) {uni}"
        base_exp = "({nom}±{err})×10{expf} {uni}"

        if modo == "repr":
            return self.__repr__()

        if rouding == "ifsc" or rouding == "-":
            n = 0
            nom = self_nom
            err = self_err
            # Arredonde o erro para a maior casa significativa
            if err != 0.0:
                while err < 1.0:
                    n -= 1
                    err *= 10
                while err >= 10.0:
                    n += 1
                    err /= 10
                err = round(err)*10**n
                # Arredonde o valor nominal de acrodo
                if n <= 0:
                    nom = round(nom, -n)
                else:
                    nom = round(nom*10**(-n))*10**n
            # Converta para string tomando cuidado com zeros desnecessários
            if err == int(err) and err != 0.0:
                err = str(int(err))
                nom = str(int(nom))
            elif err != 0.0:
                err = "{:.6f}".format(err).rstrip('0') # Isso evita com que 0.0000001 cause problemas
                if err[-1] == ".":
                    err = err + "0"
                nom = "{:.6f}".format(nom) # Isso evita com que 0.0000001 cause problemas
            else:
                err = "0"
                nom = "{}".format(nom)
            if err != 0.0:
                # Verifique se não faltam zeros no nominal
                while nom.find(".") >= 0 and err.find(".") >= 0 and len(nom)-nom.find(".") < len(err)-err.find("."):
                    nom += "0"
                # Verifique se não há zeros em excesso no nominal
                while nom.find(".") >= 0 and err.find(".") >= 0 and len(nom)-nom.find(".") > len(err)-err.find("."):
                    nom = nom[:-1]
        elif rouding == "full":
            nom = str(self_nom)
            err = str(self_err)
        else:
            raise ValueError("{} não é um parâmetro válido de arredondamento".format(rouding))
        sep = "±"
        uni = unidades_em_texto(self.unidades_originais)
        if modo == "latex":
            uni = unidades_em_texto(self.unidades_originais, estilo="latex")
            base = "({nom} \\pm {err})\\textrm{{ {uni}}}"
            base_exp = "({nom} \\pm {err})\\times10^{{{expn}}}\\textrm{{ {uni}}}"
        elif modo == "siunitx":
            uni = unidades_em_texto(self.unidades_originais, estilo="siunitx")
            base = "\\SI{{{nom}+-{err}}}{{{uni}}}"
            base_exp = "\\SI{{{nom}E{expn}+-{err}E{expn}}}{{{uni}}}"
        elif modo == "txt":
            uni = unidades_em_texto(self.unidades_originais, estilo="latex")
            base = "({nom}+/-{err}) {uni}"
            base_exp = "({nom}+/-{err})*10^{expf} {uni}"
        else:
            expf = gera_expoente(exp)

        # Prepare para imprimir
        d = {}
        d["nom"] = nom
        d["err"] = err
        d["uni"] = uni
        d["expn"] = exp
        d["expf"] = expf

        if exp == 0:
            return base.format(**d)
        else:
            return base_exp.format(**d)

        

class MCarlo(Medida):
    def __add__(self,outro):
        unidades = get_unidades(outro)
        outro = self._torne_medida(outro, True)
        if not adimensional(self):
            unidades = self.unidades_originais
        m = MCarlo((self.nominal+outro.nominal, math.sqrt(self.incerteza**2+outro.incerteza**2)), unidades)
        return m
    def __sub__(self, outro):
        unidades = get_unidades(outro)
        outro = self._torne_medida(outro, True)
        if not adimensional(self):
            unidades = self.unidades_originais
        m = MCarlo((self.nominal-outro.nominal, math.sqrt(self.incerteza**2+outro.incerteza**2)), unidades)
        return m 
    def __mul__(self, outro):
        outro = self._torne_medida(outro, False)
        multiplicacao=montecarlo(lambda x,y:x*y,self,outro)
        nom = multiplicacao.nominal
        err = multiplicacao.incerteza
        m = MCarlo((nom, err), simplifica_unidades(self.unidades_originais, outro.unidades_originais))
        return m
    def __div__(self, outro):
        outro = self._torne_medida(outro, False)
        divisao=montecarlo(lambda x,y:x/y,self,outro)
        nom = divisao.nominal
        err = divisao.incerteza
        m = MCarlo((nom, err), simplifica_unidades(self.unidades_originais, outro.unidades_originais, inverte=True))
        return m
    def __abs__(self):
        m = MCarlo((abs(self.nominal), self.incerteza), self.unidades_originais)
        return m
    def __pow__(self, outro):
        if not isinstance(outro, Medida):
            return self.__pow__(Medida(outro))
    
        if outro.dimensao != (0, 0, 0, 0, 0, 0, 0):
            raise NotImplementedError("o expoente deve ser adimensional")

        unidades = []
        if outro.nominal == int(outro.nominal) or (1/outro.nominal) == int(1/outro.nominal):
            for u in self.unidades_originais:
                unidades.append(u.nova_unidade_por_expoente(outro.nominal))
            unidades = simplifica_unidades(unidades)
        else:
            unidades = simplifica_unidades(self.unidades_originais)

        power=montecarlo(lambda x,y:x**y,self,outro)
        nom=power.nominal
        err=power.incerteza
        return MCarlo((nom,err), unidades)