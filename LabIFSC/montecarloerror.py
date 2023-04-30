from LabIFSC import*
import pkg_resources
installed= {pkg.key for pkg in pkg_resources.working_set}

def skewness(valores,desviopadrao,media):#usado para medir o quanto uma distribuição é assimétrica (ou quanto ela foge de uma gaussiana)
  m3=0
  for j in valores:
    m3+=(j-media)**3
  m3=m3/len(valores)
  g1=m3/(desviopadrao**(1.5))
  return g1

def montecarlo(funcao=sum,N=1e4,parametros=[Medida((2,0.14),"m"),Medida((4,0.1),"m")],hist=False,bins=100,assimetria="false",comparar=False):
  N=int(N)
  samples=[]
  if "numpy" in installed:
    from numpy.random import normal
    for j in range(len(parametros)): #numeros aleatorios com numpy
      samples.append(normal(parametros[j].nominal,parametros[j].incerteza,N))
  else:
    from random import gauss
    for j in range(len(parametros)): #numeros aleatorios com random 
      samples.append(gauss(parametros[j].nominal,parametros[j].incerteza,N))
  values=[]
  for k in range(N):
    temp=[]
    for j in range(len(parametros)):
      temp+=[samples[j][k]]
    try:
      values.append(funcao(tuple(temp)).nominal) #caso estejamos usando uma função do LabIFSC
    except:
      values.append(funcao(tuple(temp))) #outras funções
  if "matplotlib" in installed and hist==True:
    from matplotlib.pyplot import hist
    hist(values,bins)
  if "numpy" in installed:
    from numpy import average,std
    media=average(values)
    desviopadrao=std(values)
  else:
    media=sum(values)/len(values)
    desviopadrao=0
    for j in values:
      desviopadrao+=(media-j)**2
    desviopadrao=(desviopadrao/len(values))**(1/2)
  skew=skewness(values,desviopadrao,media)
  if assimetria ==True:
    print(f"O terceiro momento estátistico é {skew}")
  try: #tenta rodar a função recebendo uma medida como parametro
    linear=funcao(parametros)
  except:
    linear=False 
    if comparar==True: print("Função não é nativa do LabIFSC, logo a comparação não é possível")
  try: #tenta extrar a unidade
    unidade=linear.unidade()
  except:
    unidade=""
  if comparar==True and isinstance(linear, Medida): print(f"Esse é o resultado linear {linear}")
  return Medida((media,desviopadrao),unidade)
