# LabIFSC

Uma biblioteca para Python 2 e Python 3 para propagação de erro e conversão de unidades utilizando os métodos (um tanto insólitos) que os professores de lab de física do IFSC-USP insistem.

**Features diferentes da biblioteca original**
 
:ballot_box_with_check: Adicionado suporte à arrays do numpy (numpy.ndarray) na função M(). Agora, quando é passado um numpy.ndarray para M(), ela retornará um numpy.ndarray tendo objetos LabIFSC.Medida.medida como seus elementos. As operações entre arrays do numpy que multiplicam elemento-elemento estão funcionando, pois as operações elemento-elemento são definidas na classe Medida.

:white_square_button: Adicionado suporte à criação de tabelas

:ballot_box_with_check: A formatação {:latex,ifsc} é ligeiramente diferente da original. Ao printar uma medida com {:latex,ifsc}, será feito da seguinte maneira: (45,4 \pm 0,01)\textrm{ m} ou (19,4 \pm 0,03)\times10^{-3}\textrm{ m³}.

:ballot_box_with_check: Monte Carlo ,explicação mais extensa da função em um [google colab](https://colab.research.google.com/github/viniciusdutra314/LabIFSC/blob/master/MonteCarlo_examples.ipynb)

A partir das medidas $x$, guardadas na lista "parametros", variáveis aleatórias $X$ são geradas com distribuição normal Gauss(μ=x.nominal, σ=x.incerteza). Essas variáveis são calculadas $N$ vezes na função definida pelo usuário. 

É possível visualizar um histograma através da instalação da biblioteca [matplotlib](https://github.com/matplotlib/matplotlib) com o parâmetro hist=True. Além disso, é possível controlar a quantidade de bins bins=numero_de_bins. 
Com o objetivo de comparar com o método linear de incertezas,caso a função esteja definida na biblioteca original, é possível ativar o parâmetro comparar=True. Recomenda-se a instalação da biblioteca [numpy](https://github.com/numpy/numpy) para que os cálculos sejam realizados mais rapidamente, embora não seja obrigatório. 

**Exemplo**
O cálculo de uma exponencial $e^{1\pm0.5}$ é efetuado, importante lembrar que a exponencial **não** é uma função nativa do LabIFSC, e também com uma alta incerteza de 50%. Para esse cálculo, é necessário inserir uma função, a qual pode ser definida como uma expressão lambda x: math.exp(x) ou como uma função definida, por exemplo, def exponencial(x): return math.exp(x). Nesse processo, a visualização do histograma é ativada com o parâmetro hist=True, e será feita uma comparação com a biblioteca original do LabIFSC comparar=True


```python
    print(montecarlo(lambda x: math.exp(x),Medida((1,0.5),""),hist=True,comparar=True)
```
<img src="exemplomontecarlo.jpg" width="600" height="600">

#### Vantagens:
1. É importante destacar que o cálculo de incertezas muitos grandes não funciona com aproximações lineares, o que enfatiza a importância da utilização de simulações de Monte Carlo para essas situações.
2. O método estatístico adicionado a biblioteca pode ser aplicado em qualquer função cujas variáveis aleatórias X pertençam ao seu domínio, expandindo significativamente as possibilidades de funções que podem ser utilizadas na biblioteca.


# Sumário
1. [Instalação](#instalação)
    1. [PIP](#pip)
    2. [Manualmente](#manualmente)
2. [Uso](#uso)
    1. [O Básico](#o-básico)
    2. [Comparações](#comparações)
    3. [Propagação de Erro](#propagação-de-erro)
    4. [Unidades](#unidades)
    5. [Formatação de Números](#formatação-de-números)
    6. [Sequências e Tabelas](#sequências-e-tabelas)

# Instalação

## PIP

Instale a biblioteca usando o comando:

```pip install git+https://github.com/brenopelegrin/LabIFSC```

## Manualmente

Caso o PIP não esteja disponível ou não funcione, é possível utilizar a LabIFSC simplesmente colocando a pasta [```LabIFSC```](https://github.com/brenopelegrin/LabIFSC/archive/master.zip) dentro da pasta em que o seu script está. Exemplo:

```
─┬ Minha pasta qualquer
 ├─ Relatório.tex
 ├─ Relatório.pdf
 ├─ Foto1.jpg
 ┊
 ├─ meu script.py
 └┬ LabIFSC
  ├─ __init__.py
  ├─ geral.py
  ├─ lista_de_unidades.py
  ├─ matematica.py
  ├─ medida.py
  ├─ tabela.py
  └─ unidade.py
```

[Download da Última Versão (.zip)](https://github.com/brenopelegrin/LabIFSC/archive/master.zip)

[Lista de Todas as Versões](https://github.com/brenopelegrin/LabIFSC/releases)

# Uso

## O Básico

Para utilizar essa biblioteca:
```python
from LabIFSC import *
```

A principal classe é a ```Medida```. Ela pode ser inicializada de diversas formas como mostrado abaixo.
(Note que todas essas medidas são iguais, o que muda é a apenas o formato)

```python
# A unidade é sempre opcional
m = Medida((130, 27), "m")  # Medida((valor nominal, erro), unidade)
m = Medida("130+-27", "m")  # Medida(valor+-erro, unidade)
m = Medida("130+/-27", "m") # Medida(valor+/-erro, unidade)
m = Medida("130±27", "m")   # Medida(valor±erro, unidade)
m = Medida("130(27)", "m")  # Medida(valor(erro), unidade)
```

Também podemos usar a abreviatura ```M()```, a qual funciona de forma bem parecida:

```python
m = M((130, 27), "m")  # Medida((valor nominal, erro), unidade)
m = M("130+-27", "m")  # Medida(valor+-erro, unidade)
m = M("130+/-27", "m") # Medida(valor+/-erro, unidade)
m = M("130±27", "m")   # Medida(valor±erro, unidade)
m = M("130(27)", "m")  # Medida(valor(erro), unidade)
```

No entanto, ```M()``` também nos permite criar listas de medidas com facilidade:

```python
w = M(np.array([1, 40, 22, 11]), incerteza=1, unidade="m")
x = M([7, 15, 28, 42, 49, 61], incerteza=1, unidade="cm")
y = M(["1", "2", "3", "4", "4", "6"], incerteza=0.01, unidade="s")
z = M(["7+-1", "15+/-0.1", "28±10", "42(3)", 49, 61], unidade="kg", incerteza=0.01)
```

Uma instância de ```Medida``` tem os seguintes atributos:

```python
m = Medida("130±27", "ft")
print(m.nominal)      # 130.0
print(m.incerteza)    # 27.0
print(m.si_nominal)   # 39.64
print(m.si_incerteza) # 8.2296
```

Os valores prefixados com ```si_``` estão em unidades do MKS, ou seja: metro, radiano, quilograma, segundo, Kelvin, Ampère e mol. As dimensões físicas são, respectivamente: comprimento (L), ângulo (A), massa (M), tempo (T), temperatura (K), corrente (I) e "número" (N). Quando uma medida é adimensional, usa-se ∅ como mostrado no excerto abaixo:

```python
# Lembre-se que lb é o símbolo para libras/pounds
M("1") # <1.0±0.0  = 1.0±0.0 ∅>
M("1", "lb/kg") # <1.0±0.0 lb kg⁻¹ = 0.45359237±0.0 ∅>
```

Também é possível converter uma medida para sua equivalente no SI usando o método ```.SI()``` como no exemplo abaixo:

```python
M("1+-0.1", "ft").SI() # <0.3048±0.03048 m = 0.3048±0.03048 L1>
```

## Comparações

As comparações entre medidas não comparam os valores em si, mas sim verificam se as medidas são estatisticamente equivalentes.

A equação que determina se duas medidas são equivalentes é: ```|x₁ - x₂| ≤ 2(σ₁ + σ₂)```. Já a equação~que determina não equivalência é ```|x₁ - x₂| > 3(σ₁ + σ₂)```.

Cuidado deve ser tomado já que duas medidas podem não ser iguais nem diferentes ao mesmo tempo.

```python
m1 = Medida("100+/-7")
m2 = Medida("110+/-3")
m1 == m2 # True
m1 != m2 # False
```

```python
m1 = Medida("100+/-7")
m2 = Medida("125+/-3")
m1 == m2 # False
m1 != m2 # False
```

```python
m1 = Medida("100+/-7")
m2 = Medida("131+/-3")
m1 == m2 # False
m1 != m2 # True
```

## Propagação de Erro

A propagação de erro é feita automaticamente nas seguintes operções: ```+```, ```-```, ```*```, ```/```, ```**```.

```python
m1 = Medida("20+/-1")
m2 = Medida("4+/-2")
print(m1+m2) # 24±3
print(m1-m2) # 16±3
print(m1*m2) # 80±40
print(m1/m2) # 5±3
print(m1**4) # 160000±30000
print(3**m2) # 81±200
print(m1**m2) # Erro
```

Infelizmente, não há propagação de erro automática nas funções da biblioteca ```math```, porém, a LabIFSC provê as seguintes funções com propagação de erro:

```python
m1 = Medida("20+/-1.5")
print(cos(m1))     # 0±1
print(sin(m1))     # 0.9±0.6
print(tan(m1))     # 2±9
print(cot(m1))     # 0±2
print(sec(m1))     # 2±8
print(csc(m1))     # 2.2±0.7
print(arc_cos(m1)) # 0.8±0.1 rad
print(arc_sin(m1)) # 0.8±0.1 rad
print(arc_tan(m1)) # 0.6±0.1 rad
print(log(m1))     # 2.73±0.07
print(log10(m1))   # 1.30±0.03
print(log2(m1))    # 4.3±0.1
print(ln(m1))      # 3±0.08
print(sqrt(m1))    # 4.5±0.2
print(cbr(m1))     # 2.71±0.0
```

## Unidades

Todas as medidas podem ter um parâmetro adicional de unidade após o valor. Por exemplo:

```python
m1 = Medida("1+/-0.001", "m")
m2 = Medida("1+/-0.001", "ft")
m3 = Medida("1+/-0.02", "s")
```

(O arquivo ```LabIFSC/lista_de_unidades.py``` contém todas as unidades suportadas por esta biblioteca.)

As contas de soma e subtração sempre ficam nas unidades do primeiro argumento nas operações, exceto quando o primeiro argumento for adimensional. Neste caso, as unidades serão as do segundo argumento.

Já as contas de multiplicação e divisão simplesmente combinam as unidades.

```python
# Lembre-se que ft é o símbolo para pés
print(m1+m2) #  2.610±0.001 m
print(m2+m1) #  5.586±0.004 ft
print(m1-m2) # -0.610±0.001 m
print(m2-m1) # -3.586±0.004 ft
print(m1*m2) #  1.000±0.002 ft m
print(m2*m1) #  1.000±0.002 ft m
print(m1/m2) #  1.000±0.002 m ft⁻¹
print(m2/m1) #  1.000±0.002 ft m⁻¹
print(m1*m3) #  1.00±0.02 m s
print(m1/m3) #  1.00±0.02 m s⁻¹
```

Para converter as unidades:

```python
m4 = m1*m2/m3
print(m4.converta("m^2 s^-1")) # 1.610±0.007 m² s⁻¹
print(m4.converta("m^2/s"))    # 1.610±0.007 m² s⁻¹
```

## Formatação de Números

Uma mesma ```Medida``` pode ser impressa de diferentes formas:

```python
m1 = Medida("1.23456789+/-0.015", "m lb/s")

# Formatação padrão, do jeito que os profs de lab gostam
print(m1)              # 1.23±0.02 m lb s⁻¹
print(str(m1))         # 1.23±0.02 m lb s⁻¹
print("{}".format(m1)) # 1.23±0.02 m lb s⁻¹

# Representação do objeto Medida. Temos o valor original à esquerda e o valor no SI, bem como a dimensão física, à direita
print(m1.__repr__())        # <1.23456789±0.015 m lb s⁻¹ = 0.5599905751509993±0.00680388555 L1M1T-1>
print("{:repr}".format(m1)) # <1.23456789±0.015 m lb s⁻¹ = 0.5599905751509993±0.00680388555 L1M1T-1>

# Diferentes estilos de representação com o arredondamento padrão (arredondamento ifsc)
print("{}".format(m1))          # 1.23±0.02 m lb s⁻¹
print("{:-}".format(m1))        # 1.23±0.02 m lb s⁻¹
print("{:latex}".format(m1))    # 1.23\pm0.02\textrm{ m lb s^-1}
print("{:siunitex}".format(m1)) # 1.23±0.02 m lb s⁻¹
print("{:txt}".format(m1))      # 1.23+/-0.02 m lb s^-1

# Diferentes estilos de representação com o arredondamento do ifsc
print("{}".format(m1))               # 1.23±0.02 m lb s⁻¹
print("{:-,ifsc}".format(m1))        # 1.23±0.02 m lb s⁻¹
print("{:latex,ifsc}".format(m1))    # (1.23 \\pm 0.02) \\textrm{ m lb s^-1}
print("{:siunitex,ifsc}".format(m1)) # 1.23±0.02 m lb s⁻¹
print("{:txt,ifsc}".format(m1))      # 1.23+/-0.02 m lb s^-1

# Diferentes estilos de representação sem arredondamento
print("{:-,full}".format(m1))        # 1.23456789±0.015 m lb s⁻¹
print("{:latex,full}".format(m1))    # 1.23456789\\pm0.015\\textrm{ m lb s^-1}
print("{:siunitex,full}".format(m1)) # 1.23456789±0.015 m lb s⁻¹
print("{:txt,full}".format(m1))      # 1.23456789+/-0.015 m lb s^-1

# Diferentes estilos de representação com o arredondamento do ifsc usando 10^-2 como base
print("{:-,ifsc,-2}".format(m1))        # (123±2)×10⁻² m lb s⁻¹
print("{:latex,ifsc,-2}".format(m1))    # (123\pm2)\cdot10^{-2}\textrm{ m lb s^-1}
print("{:siunitex,ifsc,-2}".format(m1)) # (123±2)×10⁻² m lb s⁻¹
print("{:txt,ifsc,-2}".format(m1))      # (123+/-2)*10^-2 m lb s^-1

# Diferentes estilos de representação sem arredondamento usando 10^-2 como base
print("{:-,full,-2}".format(m1))        # (123.45678899999999±1.5)×10⁻² m lb s⁻¹
print("{:latex,full,-2}".format(m1))    # (123.45678899999999\pm1.5)\cdot10^{-2}\textrm{ m lb s^-1}
print("{:siunitex,full,-2}".format(m1)) # (123.45678899999999±1.5)×10⁻² m lb s⁻¹
print("{:txt,full,-2}".format(m1))      # (123.45678899999999+/-1.5)*10^-2 m lb s^-1
```

## Sequências e Tabelas

Essa biblioteca provê funções para calcular média, desvio padrão e linearização de dados. Mais detalhes estão no exemplo abaixo:

```python
x = M(["147.0", "161.8", "174.6", "161.01", "175.6", "166.0"], incerteza=0.1, unidade="cm")
print(media(x))                             # 160±10 cm
print(media(x, incerteza="desvio padrão"))  # 160±10 cm
print(media(x, incerteza="propagação"))     # 164.3±0.1 cm
print(desvio_padrao(x))                     # 10.516907815513074

x = M(["7", "15", "28", "42", "49", "61"], incerteza=1, unidade="cm")
y = M(["1", "2", "3", "4", "4", "6"], incerteza=0.01, unidade="s")
linearize(x, y, imprimir=True)
x[1] = x[1].converta("m")
x[2] = x[2].converta("km")
linearize(x, y, imprimir=True)
# Note que, mesmo com unidades diferentes, as linearizações tem o mesmo resultado
```
