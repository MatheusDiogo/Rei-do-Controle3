# Rei-do-Controle3

Programa com interface gráfica para projetar um controlador digital Gc(z) a partir de uma planta G(s).

## Uso

Execute `dist\ReiDoControle.exe` (ou `python app.py`).

1. **G(s)**: digite a função inteira, por exemplo `2.5537e+06 / (180.53*s^3 + 26.873*s^2 + s)`
   ou `10*(s+2)/((s+1)*(s+5))`. A multiplicação é **somente** com `*` (ex: `10*s`, `(s+1)*(s+2)`); o `.` é sempre separador decimal. Para potência, use `^`.
   Atraso de transporte: multiplique por `e^(-Td*s)` (ou `exp(-Td*s)`), ex: `15.436*e^(-0.82424*s)/(8.7953*s + 1)`.
   A discretização ZOH é exata (z^-d + atraso fracionário pela Z modificada, como o `c2d` do MATLAB com InputDelay).
2. **T**: período de amostragem em **milissegundos** (ex: 100 = 0,1 s).
   A planta também pode ser informada já discretizada, em `z`: `(49.75*z + 44.97)/(z^2 - 0.852*z - 0.05611)`
   (o RF exige G(s)).
3. **Requisitos**: overshoot (%), tempo de pico (s) e tempo de acomodação 2% (s). Deixe em branco o que não for pedido.
   Marque **Erro estacionário nulo** para garantir erro zero ao degrau.
   **Amplitude do degrau (UE)** escala a resposta e o esforço de controle. **Limite de |u| (V)** é verificado em
   todos os métodos e entra como restrição na otimização.
   **Setpoint do TIA (opcional)**: um .txt com uma amostra por linha, espaçadas de T, como copiado do TIA
   (ex.: [exemplos/setpoint_onda_quadrada.txt](exemplos/setpoint_onda_quadrada.txt)). A otimização passa a minimizar
   Σ(yd - y)² sobre esse r(k), com o |u| real. Todos os métodos mostram a resposta, o u e a SomaErro para ele.
   yd(k) também pode vir de um arquivo; sem ele, yd = T(z) desejada aplicada a r(k).
4. **Método**:
   - **LR / Cancelamento**: o zero do controlador cancela o polo mais lento da planta.
   - **LR / Bissetriz**: zero e polo posicionados pela bissetriz em torno do polo desejado zd.
   - **LR / Avanço de fase**: zero diretamente abaixo de zd (zc = Re(zd)) e polo pela condição de ângulo.
   - **RF**: avanço de fase pelo diagrama de Bode (MF e ωc tirados dos requisitos), discretizado por Tustin.
   - **Síntese de FT**: T(z) desejada de 2ª ordem (polos zd, T(1) = 1 para erro nulo e atraso z^-k se for preciso
     para ser causal), com Gc = T/[G·(1-T)]. Zeros da planta fora do círculo unitário ficam em T(z). Há também uma variante
     que mantém em T(z) os zeros em z <= 0, para evitar ringing em u.
   - **Otimização (ISE)**: Gc de 1ª ou 2ª ordem (com integrador fixo se o erro nulo estiver marcado). Os coeficientes
     minimizam Σ(yd - y)², com estabilidade e |u| <= limite como restrições. A busca global é feita por evolução diferencial
     (equivale ao Solver com início múltiplo). "Suavizar u" evita polos do Gc com parte real negativa.
5. Clique em **CALCULAR**. A aba *Resultado* mostra Gc(z) com ganho, zeros, polos, equação de diferenças,
   coeficientes em z^-1 e as especificações alcançadas, incluindo o esforço de controle máximo |u|.
   Se Gc(z) passar de 2ª ordem, ele também é dividido em blocos em cascata de até 2ª ordem (Gc1·Gc2·...),
   cada um com sua equação de diferenças, prontos para blocos de 2ª ordem do TIA Portal.
   O gráfico de esforço mostra u = T(z)/G(z) para um degrau unitário na referência. Os gráficos (LGR/Bode e degrau) e a memória de cálculo completa ficam nas outras abas.

### Blocos GZ1/GZ2 do TIA

Com a opção **Compatível com blocos GZ1/GZ2 do TIA (matched)** marcada (padrão), a aba *Resultado* começa com os
valores para preencher em cada bloco:

- **GZ1**: G(s) = (c0·s + c1)/(s + d1), com **c0, c1, d1**.
- **GZ2**: G(s) = (c0·s² + c1·s + c2)/(s² + d1·s + d2), com **c0, c1, c2, d1, d2**.

Esses c/d são o *matched inverso* de cada bloco de Gc(z) (s = ln(z)/Ts). Assim o `c2d(...,'matched')` feito pelo bloco
reproduz exatamente o controlador projetado. Para conferir, também são mostrados os b0, b1, b2, a1, a2 que o bloco deve calcular.
Nesse modo:

- o integrador do LR passa a ser 1/(z-1) (equivale a 1/s);
- o RF passa a usar matched em vez de Tustin;
- variantes com raiz real em z <= 0, que não têm G(s) equivalente, são evitadas.

O projeto clássico é sempre tentado primeiro. Se ele for geometricamente inviável ou deixar a malha fechada
instável, o programa avalia variantes e usa a que melhor atende aos requisitos, avisando qual foi:
bissetriz com centro deslocado ou com 2–3 estágios, cancelamento dos 2 polos mais lentos e diferentes posições do zero do PI.

## Gerar o executável

```
build.bat
```

Arquivos: `controle.py` (cálculos: c2d, feedback, stepinfo, margin e os métodos de projeto) e `app.py` (interface).
