from .medida import Medida

def montecarlo(func, *args, comparar=False, N=10_000, hist=False, bins=100, probabilidade=False):
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
     Medida(media,desviopadrao,unidade)
  Raises:
    func e/ou args não existem ValueError("Não foi especificado uma função ou seus parametros")
    um dos args não é uma medida ValueError("Todos os parametros precisam ser medidas")
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
        if not isinstance(j, Medida): raise ValueError("Todos os parametros precisam ser medidas")
    if not isinstance(N, int) or not isinstance(bins, int):
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
    return Medida((media, desviopadrao), unidade)
