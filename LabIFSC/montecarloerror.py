from sys import modules
instalados=[]
if "matplotlib.pyplot" in modules: instalados.append("matplot")
if "numpy" in modules: instalados.append("numpy")

def skewness(*args):
  '''usado para medir o quanto uma distribuição é assimétrica (ou quanto ela foge de uma gaussiana)
  para uma gaussiana, o skewness é zero, para um skewness alto (~1) é importante olhar para o histograma
  porque a distribuição de probabilidade não sera gaussiana, logo a interpretação da média e desvio padrão
  será diferente.
  args[0] são os valores, args[1]=media e args[2]=desviopadrão, os dois ultimos são opicionais sendo
  usados no caso desses numeros já terem sido calculados de forma a acelerar o código

  '''
  valores=args[0]
  if len(args)==1:
    media=sum(valores)/(tamanho:=len(valores))
    desviopadrao=0
    for j in valores: desviopadrao+=(media-j)**2
    desviopadrao=(desviopadrao/tamanho)**(1/2)  
  else:
    media=args[1]
    desviopadrao=args[2]
    tamanho=len(args[0])
  skew=0
  for j in valores:
    skew+=(j-media)**3
  skew=skew/tamanho
  skew=skew/(desviopadrao**(1.5))
  return skew



def montecarlo(*args,hist=False,assimetria=False,comparar=False,bins=100,N=10_000):
  '''A partir das medidas ,guardadas na lista "parametros", variáveis aleatórias são geradas 
  com distribuição normal Gauss(μ=x.nominal, σ=x.incerteza). Essas variáveis são calculadas N
  vezes na função definida pelo usuário. É possível visualizar um histograma através da instalação 
  da biblioteca matplotlib com o parâmetro hist=True. Além disso, é possível controlar a quantidade 
  de bins e requisitar o terceiro momento estatístico da distribuição, que mede a assimetria da função, 
  utilizando o parâmetro assimetria=True. Com o objetivo de comparar com o método linear de incertezas,
  caso a função esteja definida na biblioteca original, é possível ativar o parâmetro comparar=True. 
  Recomenda-se a instalação da biblioteca numpy para que os cálculos sejam realizados mais rapidamente, 
  embora não seja obrigatório.
'''
  #importando variaveis e mensagens de erro
  try:
    funcao=args[0]
    parametros=args[1::]
  except: raise ValueError("Não foi especificado uma função ou seus parametros")
  for j in parametros:
    assert isinstance(j,Medida) , "Todos os parametros precisam ser medidas"
  N,bins=int(N),int(bins) 
  #criando números aleatorios
  amostras=[]
  if "numpy" in instalados:
    from numpy.random import normal
    for j in range(len(parametros)): #numeros aleatorios com numpy
      amostras.append((normal(parametros[j].nominal,parametros[j].incerteza,N)))
  else:
    from random import gauss
    for j in range(len(parametros)): #numeros aleatorios com random 
      amostras.append(gauss(parametros[j].nominal,parametros[j].incerteza))
  valores=[]
  for k in range(N):
    temp=[]
    for j in range(len(parametros)):
      temp.append(amostras[j][k])
    try:
      valores.append(funcao(*temp).nominal) #caso estejamos usando uma função do LabIFSC
    except:
      valores.append(funcao(*temp)) #funções não nativas do LabIFSC
  if "matplot" in instalados and hist==True:
    from matplotlib.pyplot import hist
    hist(valores,bins)
  if "numpy" in instalados: #média e desvio-padrão usando numpy
    from numpy import average,std
    media=average(valores)
    desviopadrao=std(valores)
  else: #usando somente python
    media=sum(valores)/len(valores)
    desviopadrao=0
    for j in valores: desviopadrao+=(media-j)**2
    desviopadrao=(desviopadrao/len(valores))**(1/2)  
  if assimetria ==True:
    skew=skewness(valores,media,desviopadrao)
    print(f"O terceiro momento estátistico (skewness) é {skew}")
  try: #tenta rodar a função recebendo uma medida como parametro
    linear=funcao(*parametros)
  except:
    linear=False 
    if comparar==True: print("Função não é nativa do LabIFSC, logo a comparação não é possível")
  try: #tenta extrar a unidade
    unidade=linear.unidade()
  except:
    unidade=""
  if comparar==True and isinstance(linear, Medida): print(f"Esse é o resultado linear {linear}")
  return Medida((media,desviopadrao),unidade)
