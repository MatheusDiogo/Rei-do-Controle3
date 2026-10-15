"""
Núcleo de cálculo do Rei do Controle.

Projeto de controladores digitais Gc(z) a partir de uma planta G(s):
  - Lugar das Raízes (LR): Cancelamento do polo mais lento ou Bissetriz
  - Resposta em Frequência (RF): avanço de fase clássico (Bode) + Tustin

Baseado nos templates MATLAB Projeto_Bissetriz_SC3.m e Projeto_RF_SC3.m.
Usa apenas numpy/scipy (equivalentes de tf, c2d, feedback, stepinfo, margin).
"""
import ast
import math

import numpy as np
from scipy import signal
from scipy.linalg import expm

TOL_INTEGRADOR_S = 1e-6   # |p| abaixo disso em s é tratado como integrador
TOL_INTEGRADOR_Z = 1e-6   # |p - 1| abaixo disso em z é tratado como integrador


class ErroProjeto(Exception):
    """Erro de entrada ou de projeto, com mensagem para o usuário."""


# =========================================================================
# Polinômios / funções de transferência
# =========================================================================
def _limpa(p):
    """Remove zeros à esquerda (coeficientes desprezíveis) de um polinômio."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    if p.size == 0:
        return np.array([0.0])
    escala = np.max(np.abs(p))
    if escala == 0:
        return np.array([0.0])
    idx = np.argmax(np.abs(p) > escala * 1e-14)
    return p[idx:]


def _soma(a, b):
    n = max(len(a), len(b))
    return np.pad(a, (n - len(a), 0)) + np.pad(b, (n - len(b), 0))


class Racional:
    """Função racional em s (num/den) com atraso de transporte e^(-atraso·s), usada pelo parser de G(s)."""

    def __init__(self, num, den=(1.0,), atraso=0.0):
        self.num = _limpa(num)
        self.den = _limpa(den)
        self.atraso = float(atraso)

    @staticmethod
    def _conv(x):
        return x if isinstance(x, Racional) else Racional([float(x)])

    def __add__(self, o):
        o = self._conv(o)
        if not np.any(o.num):
            return self
        if not np.any(self.num):
            return o
        if abs(self.atraso - o.atraso) > 1e-12:
            raise ErroProjeto("O atraso e^(-Td*s) deve multiplicar a G(s) inteira "
                              "(não pode somar termos com atrasos diferentes).")
        return Racional(_soma(np.polymul(self.num, o.den), np.polymul(o.num, self.den)),
                        np.polymul(self.den, o.den), self.atraso)

    __radd__ = __add__

    def __neg__(self):
        return Racional(-self.num, self.den, self.atraso)

    def __sub__(self, o):
        return self + (-self._conv(o))

    def __rsub__(self, o):
        return self._conv(o) - self

    def __mul__(self, o):
        o = self._conv(o)
        return Racional(np.polymul(self.num, o.num), np.polymul(self.den, o.den), self.atraso + o.atraso)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._conv(o)
        if not np.any(o.num):
            raise ErroProjeto("Divisão por zero na expressão de G(s).")
        return Racional(np.polymul(self.num, o.den), np.polymul(self.den, o.num), self.atraso - o.atraso)

    def __rtruediv__(self, o):
        return self._conv(o) / self

    def __pow__(self, n):
        if not float(n).is_integer():
            raise ErroProjeto("Expoentes em G(s) devem ser inteiros.")
        n = int(n)
        r = Racional([1.0])
        for _ in range(abs(n)):
            r = r * self
        return r if n >= 0 else Racional([1.0]) / r


def _exponencial(x):
    """e^(x): número comum, ou atraso de transporte quando x = -Td·s."""
    if not isinstance(x, Racional):
        return math.exp(x)
    num, den = _limpa(x.num), _limpa(x.den)
    linear = (len(den) == 1 and len(num) == 2 and abs(num[1]) <= 1e-12 * abs(num[0]) and x.atraso == 0)
    if not linear:
        raise ErroProjeto("Exponencial em G(s) só é aceita como atraso de transporte: e^(-Td*s), "
                          "por exemplo e^(-0.82424*s).")
    a = num[0] / den[0]
    if a > 0:
        raise ErroProjeto("e^(+Td*s) é um avanço no tempo (não causal). Use e^(-Td*s).")
    return Racional([1.0], [1.0], -a)


def _avalia_ast(no):
    if isinstance(no, ast.Expression):
        return _avalia_ast(no.body)
    if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
        return float(no.value)
    if isinstance(no, ast.Num):  # Python < 3.8
        return float(no.n)
    if isinstance(no, ast.Name):
        if no.id == "s":
            return Racional([1.0, 0.0])
        if no.id == "e":
            raise ErroProjeto("Use 'e' apenas como e^(-Td*s) para o atraso de transporte.")
        raise ErroProjeto(f"Variável desconhecida em G(s): '{no.id}'. Use apenas 's'.")
    if isinstance(no, ast.UnaryOp) and isinstance(no.op, (ast.USub, ast.UAdd)):
        v = _avalia_ast(no.operand)
        return -v if isinstance(no.op, ast.USub) else v
    if isinstance(no, ast.BinOp):
        if isinstance(no.op, ast.Pow) and isinstance(no.left, ast.Name) and no.left.id == "e":
            return _exponencial(_avalia_ast(no.right))
        a, b = _avalia_ast(no.left), _avalia_ast(no.right)
        if isinstance(no.op, ast.Add):
            return a + b
        if isinstance(no.op, ast.Sub):
            return a - b
        if isinstance(no.op, ast.Mult):
            return a * b
        if isinstance(no.op, ast.Div):
            if not isinstance(a, Racional) and not isinstance(b, Racional):
                return a / b
            return Racional._conv(a) / b
        if isinstance(no.op, ast.Pow):
            if isinstance(b, Racional):
                raise ErroProjeto("O expoente em G(s) deve ser um número.")
            return Racional._conv(a) ** b if isinstance(a, Racional) else a ** b
    if isinstance(no, ast.Attribute):
        raise ErroProjeto("O '.' em G(s) é só separador decimal. Para multiplicar use '*', "
                          "por exemplo 180.53*s^3.")
    if isinstance(no, ast.Call):
        if isinstance(no.func, ast.Name) and no.func.id == "exp" and len(no.args) == 1:
            return _exponencial(_avalia_ast(no.args[0]))
        raise ErroProjeto("Falta '*' entre fatores em G(s). Use, por exemplo, (s+1)*(s+2) ou 2*(s+1).")
    raise ErroProjeto("Expressão de G(s) não reconhecida.")


def parse_planta(texto):
    """
    Converte um texto como
        "G(s) = 15.436 * e^(-0.82424*s) / (8.7953*s + 1)"
    em (num, den, Td) com den mônico e Td = atraso de transporte (s).
    Multiplicação somente com '*'; potência com '^' ou '**'; '.' é sempre
    separador decimal (vírgula também é aceita como decimal).
    Atraso: e^(-Td*s) ou exp(-Td*s), multiplicando a G(s).
    """
    t = texto.strip()
    if "=" in t:
        t = t.split("=", 1)[1]
    t = t.replace("^", "**").replace(",", ".")
    t = t.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    if not t.strip():
        raise ErroProjeto("Digite a função de transferência G(s).")
    try:
        arvore = ast.parse(t.strip(), mode="eval")
    except SyntaxError:
        raise ErroProjeto(f"Não consegui interpretar G(s): '{texto}'.\n\n"
                          "Use '*' para toda multiplicação, por exemplo: 10*s, 2*(s+1), (s+1)*(s+2).")
    r = Racional._conv(_avalia_ast(arvore))
    num, den = _limpa(r.num), _limpa(r.den)
    if not np.any(num):
        raise ErroProjeto("G(s) é identicamente zero.")
    if len(num) > len(den):
        raise ErroProjeto("G(s) é imprópria (grau do numerador maior que o do denominador).")
    if r.atraso < -1e-12:
        raise ErroProjeto("O atraso total de G(s) é negativo (não causal).")
    k = den[0]
    return num / k, den / k, max(r.atraso, 0.0)


# =========================================================================
# Utilidades de sistemas
# =========================================================================
def c2d(num, den, T, metodo="zoh"):
    """Equivalente ao c2d do MATLAB para FT SISO ('zoh' ou 'tustin')."""
    m = "bilinear" if metodo == "tustin" else metodo
    nd, dd, _ = signal.cont2discrete((num, den), T, method=m)
    nd = np.atleast_2d(nd)[0]
    k = dd[0]
    return _limpa(nd / k), _limpa(dd / k)


def c2d_zoh_atraso(num, den, T, Td):
    """
    Discretização ZOH exata de G(s)·e^(-Td·s) (equivale ao c2d do MATLAB com InputDelay).
    Td = d·T + θ (0 <= θ < T): o atraso inteiro vira z^-d e o fracionário θ é
    tratado pela transformada Z modificada (via espaço de estados).
    Retorna (num_z, den_z, d, θ).
    """
    if Td <= 0:
        nz, dz = c2d(num, den, T, "zoh")
        return nz, dz, 0, 0.0
    d = int(math.floor(Td / T + 1e-9))
    theta = Td - d * T
    if theta < 1e-9 * T:
        theta = 0.0
    z_d = np.zeros(d + 1)
    z_d[0] = 1.0                                   # z^d
    if theta == 0.0:
        nz, dz = c2d(num, den, T, "zoh")
        return nz, _limpa(np.polymul(dz, z_d)), d, 0.0
    A, B, C, D = signal.tf2ss(num, den)
    n = A.shape[0]
    D = float(np.atleast_1d(D).ravel()[0]) if np.size(D) else 0.0
    if n == 0:                                     # ganho puro: y[k] = D·u[k-d-1]
        return np.array([D]), _limpa(np.polymul([1.0, 0.0], z_d)), d, theta

    def phi_gamma(t):
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A * t
        M[:n, n:] = B * t
        E = expm(M)
        return E[:n, :n], E[:n, n:]

    Phi, _ = phi_gamma(T)
    Pt, G0 = phi_gamma(T - theta)
    _, Gth = phi_gamma(theta)
    G1 = Pt @ Gth
    N0, den_phi = signal.ss2tf(Phi, G0, C, np.zeros((1, 1)))
    N1, _ = signal.ss2tf(Phi, G1, C, np.zeros((1, 1)))
    N0, N1, den_phi = np.atleast_2d(N0)[0], np.atleast_2d(N1)[0], np.asarray(den_phi, float)
    # y[k] = C x[k] + D·u[k-1];  x[k+1] = Φx[k] + Γ0·u[k] + Γ1·u[k-1]
    num_z = _soma(_soma(np.polymul(N0, [1.0, 0.0]), N1), D * den_phi)
    den_z = np.polymul(np.polymul(den_phi, [1.0, 0.0]), z_d)
    k = den_z[0]
    return _limpa(num_z / k), _limpa(den_z / k), d, theta


def pade(Td, n=3):
    """Aproximação de Padé [n/n] de e^(-Td·s). Retorna (num, den)."""
    c = [math.factorial(2 * n - k) * math.factorial(n)
         / (math.factorial(2 * n) * math.factorial(k) * math.factorial(n - k)) for k in range(n + 1)]
    num = np.array([c[k] * (-Td) ** k for k in range(n, -1, -1)])
    den = np.array([c[k] * Td ** k for k in range(n, -1, -1)])
    return num / den[0], den / den[0]


def avalia(num, den, x):
    return np.polyval(num, x) / np.polyval(den, x)


def malha_fechada(num, den):
    """feedback(L, 1) para L = num/den."""
    n = max(len(num), len(den))
    numP = np.pad(num, (n - len(num), 0))
    denP = np.pad(den, (n - len(den), 0))
    return _limpa(numP), _limpa(denP + numP)


def _wrap(a):
    """wrapToPi."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def ganho_dc_discreto(num, den):
    d = np.polyval(den, 1.0)
    if abs(d) < 1e-14:
        return np.inf
    return np.polyval(num, 1.0) / d


def ganho_dc_continuo(num, den):
    d = den[-1]
    if abs(d) < 1e-300:
        return np.inf
    return num[-1] / d


def resposta_degrau_discreta(num, den, T, n_min=60):
    """Resposta ao degrau de H(z)=num/den. Retorna (t, y)."""
    polos = np.roots(den) if len(den) > 1 else np.array([])
    rmax = np.max(np.abs(polos)) if polos.size else 0.0
    if rmax < 1:
        n = int(np.ceil(np.log(1e-4) / np.log(max(rmax, 1e-3)))) * 2 if rmax > 0 else n_min
        n = int(np.clip(n, n_min, 20000))
    else:
        n = 200
    b = np.pad(num, (len(den) - len(num), 0))
    y = signal.lfilter(b, den, np.ones(n))
    return np.arange(n) * T, y


def resposta_degrau_continua(num, den, t_final):
    t = np.linspace(0, t_final, 2000)
    sysc = signal.lti(num, den)
    t, y = signal.step(sysc, T=t)
    return t, y


def stepinfo(t, y, y_final, faixa=0.02):
    """Equivalente ao stepinfo (sobressinal, tp, ts 2%, tr 10-90%)."""
    info = {"Overshoot": np.nan, "PeakTime": np.nan, "SettlingTime": np.nan,
            "RiseTime": np.nan, "Peak": np.nan}
    if not np.isfinite(y_final) or abs(y_final) < 1e-12 or not np.all(np.isfinite(y)):
        return info
    ys = y / y_final
    ip = int(np.argmax(ys))
    info["Peak"] = y[ip]
    info["PeakTime"] = t[ip]
    info["Overshoot"] = max(0.0, (ys[ip] - 1.0) * 100.0)
    fora = np.where(np.abs(ys - 1.0) > faixa)[0]
    if fora.size == 0:
        info["SettlingTime"] = 0.0
    elif fora[-1] + 1 < len(t):
        info["SettlingTime"] = t[fora[-1] + 1]
    i10 = np.argmax(ys >= 0.1)
    i90 = np.argmax(ys >= 0.9)
    if ys[i90] >= 0.9 and ys[i10] >= 0.1:
        info["RiseTime"] = t[i90] - t[i10]
    return info


def fase_continua(num, den, w, atraso=0.0):
    """Fase (rad) de num/den em jw, contínua em w (soma das fases de cada fator)."""
    jw = 1j * np.atleast_1d(w)
    fase = np.zeros(jw.shape)
    for z in (np.roots(num) if len(num) > 1 else []):
        fase += np.angle(jw - z)
    for p in (np.roots(den) if len(den) > 1 else []):
        fase -= np.angle(jw - p)
    if num[0] / den[0] < 0:
        fase -= np.pi
    return fase - w * atraso


def margens(num, den, atraso=0.0, w=None):
    """Margens de ganho e fase (como margin do MATLAB) sobre uma grade densa."""
    if w is None:
        w = np.logspace(-4, 5, 40000)
    mag = np.abs(avalia(num, den, 1j * w))
    fase = fase_continua(num, den, w, atraso)
    res = {"Gm_dB": np.inf, "Wcg": np.nan, "Pm": np.inf, "Wcp": np.nan}
    lm = np.log(mag)
    idx = np.where(np.diff(np.sign(lm)) != 0)[0]
    if idx.size:
        i = idx[0]
        f = lm[i] / (lm[i] - lm[i + 1])
        wc = w[i] * (w[i + 1] / w[i]) ** f
        res["Wcp"] = wc
        res["Pm"] = math.degrees(fase_continua(num, den, np.array([wc]), atraso)[0]) + 180.0
    # cruzamento de fase em -180 (+ k*360)
    g = (fase + np.pi) / (2 * np.pi)
    idx = np.where(np.diff(np.floor(g)) != 0)[0]
    if idx.size:
        i = idx[0]
        res["Wcg"] = w[i]
        res["Gm_dB"] = -20 * np.log10(mag[i])
    return res


# =========================================================================
# Requisitos
# =========================================================================
def processa_requisitos(PO, tp, ts):
    """Obtém zeta e wn a partir de PO (%), tp (s) e ts 2% (s). Campos None são ignorados."""
    notas = []
    if PO is not None:
        if not (0 < PO < 100):
            raise ErroProjeto("O overshoot deve estar entre 0 e 100 % (exclusive).")
        L = math.log(PO / 100.0)
        zeta = abs(L) / math.sqrt(math.pi ** 2 + L ** 2)
        cand = []
        if tp is not None:
            if tp <= 0:
                raise ErroProjeto("O tempo de pico deve ser positivo.")
            cand.append((math.pi / (tp * math.sqrt(1 - zeta ** 2)), "tempo de pico"))
        if ts is not None:
            if ts <= 0:
                raise ErroProjeto("O tempo de acomodação deve ser positivo.")
            cand.append((4.0 / (zeta * ts), "tempo de acomodação"))
        if not cand:
            raise ErroProjeto("Informe também o tempo de pico e/ou o tempo de acomodação.")
        wn, origem = max(cand)
        if len(cand) == 2:
            notas.append(f"wn definido pelo {origem} (o requisito mais exigente).")
    elif tp is not None and ts is not None:
        if tp <= 0 or ts <= 0:
            raise ErroProjeto("Os tempos devem ser positivos.")
        sigma, wd = 4.0 / ts, math.pi / tp
        wn = math.hypot(sigma, wd)
        zeta = sigma / wn
        notas.append("Sem overshoot informado: zeta obtido de tp e ts.")
    else:
        raise ErroProjeto("Informe o overshoot e um tempo (tp ou ts), ou então tp e ts.")
    wd = wn * math.sqrt(1 - zeta ** 2)
    return {
        "zeta": zeta, "wn": wn, "wd": wd, "sigma": zeta * wn,
        "PO": math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2)) * 100,
        "tp": math.pi / wd, "ts": 4.0 / (zeta * wn), "notas": notas,
    }


# =========================================================================
# Projeto
# =========================================================================
MARGEM_ANG = math.radians(3.0)   # folga geométrica mínima para ângulos de zero/polo


def _polos_integradores_z(den):
    p = np.roots(den) if len(den) > 1 else np.array([])
    return int(np.sum(np.abs(p - 1.0) < TOL_INTEGRADOR_Z))


def _polos_integradores_s(den):
    p = np.roots(den) if len(den) > 1 else np.array([])
    return int(np.sum(np.abs(p) < TOL_INTEGRADOR_S))


def minreal(num, den, tol=1e-7):
    """Cancela raízes comuns entre numerador e denominador."""
    num, den = _limpa(num), _limpa(den)
    if len(num) < 2 or len(den) < 2:
        return num, den
    zs, ps = list(np.roots(num)), list(np.roots(den))
    kz, kp = num[0], den[0]
    mudou = False
    for z in list(zs):
        for p in ps:
            if abs(z - p) < tol * max(1.0, abs(p)):
                zs.remove(z)
                ps.remove(p)
                mudou = True
                break
    if not mudou:
        return num, den
    n = kz * np.real(np.poly(zs)) if zs else np.array([kz])
    d = kp * np.real(np.poly(ps)) if ps else np.array([kp])
    return n / d[0], d / d[0]


def _angulo_para_real(zd, ang):
    """Ponto x do eixo real tal que ∠(zd - x) = ang (0 < ang < π)."""
    return zd.real - zd.imag / math.tan(ang)


def _custo(info, estavel, req_in, req, complexidade):
    """Quanto o desempenho obtido se afasta dos requisitos (menor = melhor)."""
    if not estavel or not np.isfinite(info["SettlingTime"]):
        return np.inf
    c = 0.05 * complexidade
    # sem ts informado, usa o ts implícito (4/ζωn) como referência de rapidez
    ts_ref = req_in["ts"] if req_in["ts"] is not None else req["ts"]
    for chave, alvo in (("Overshoot", req_in["PO"]), ("PeakTime", req_in["tp"]),
                        ("SettlingTime", ts_ref)):
        if alvo is None:
            continue
        if chave == "PeakTime" and info["Overshoot"] < 0.5:
            continue    # resposta sem pico: tp não se aplica (ts já mede a rapidez)
        v = info[chave]
        if not np.isfinite(v):
            c += 10
            continue
        r = v / alvo
        c += 3 * max(0.0, r - 1.0) + 0.2 * abs(r - 1.0)
    return c


def _avalia_candidato(R, cand):
    """Fecha a malha (plano z) para um candidato e calcula desempenho."""
    T = R["T"]
    numL = _limpa(np.polymul(cand["num_c"], R["num_z"]))
    denL = _limpa(np.polymul(cand["den_c"], R["den_z"]))
    num_mf, den_mf = malha_fechada(numL, denL)
    polos = np.roots(den_mf) if len(den_mf) > 1 else np.array([])
    estavel = bool(np.any(numL)) and bool(np.all(np.abs(polos) < 1 - 1e-9))
    dc = ganho_dc_discreto(num_mf, den_mf) if estavel else np.nan
    t, y = resposta_degrau_discreta(num_mf, den_mf, T)
    info = stepinfo(t, y, dc) if estavel else stepinfo(t, y, np.nan)
    cand.update({
        "num_L": numL, "den_L": denL, "num_mf": num_mf, "den_mf": den_mf,
        "polos_mf": polos, "estavel": estavel, "dc": dc, "t": t, "y": y, "info": info,
        "custo": _custo(info, estavel, R["req_in"], R["req"], cand.get("complexidade", 0)),
    })
    if R.get("tia") and "zp" in cand and not representavel_matched(cand["zp"]):
        cand["custo"] += 5.0     # raiz real em z <= 0: não cabe nos blocos GZ (matched)
        cand["rotulo"] += " [fora dos blocos GZ]"
    return cand


def projeta(G_texto, T, PO, tp, ts, erro_nulo, metodo, sub_metodo="bissetriz", folga=5.0, tia=False):
    """
    Executa o projeto completo.
      metodo: 'LR' ou 'RF'
      sub_metodo (LR): 'cancelamento', 'bissetriz' ou 'avanco'
      folga (RF): graus extras de margem de fase
      tia: controlador implementável nos blocos GZ1/GZ2 do TIA (que fazem c2d 'matched'):
           integrador 1/(z-1) no LR, variantes com raiz real em z <= 0 penalizadas
           e discretização matched (em vez de Tustin) no RF
    O projeto clássico é sempre o primeiro candidato; se ele falhar (instável ou
    geometricamente inviável), variantes são avaliadas e a melhor é escolhida.
    Retorna um dicionário com tudo que a interface precisa exibir.
    """
    if T is None or T <= 0:
        raise ErroProjeto("Informe um período de amostragem T > 0.")
    num_s, den_s, Td = parse_planta(G_texto)
    req = processa_requisitos(PO, tp, ts)
    avisos = list(req["notas"])
    num_z, den_z, d_atraso, theta = c2d_zoh_atraso(num_s, den_s, T, Td)
    passos = []
    if Td > 0:
        passos.append(f"Atraso de transporte Td = {Td:.6g} s = {d_atraso}·T + {theta:.6g} s  ->  "
                      f"G(z) inclui z^-{d_atraso}" + (" e o atraso fracionário (Z modificada)" if theta else ""))
        if Td > req["ts"] / 2:
            avisos.append(f"O atraso Td = {Td:g} s é grande perto do ts desejado ({req['ts']:.3g} s): "
                          "os requisitos podem ser inatingíveis.")

    R = {
        "T": T, "num_s": num_s, "den_s": den_s, "Td": Td, "num_z": num_z, "den_z": den_z,
        "d_atraso": d_atraso, "theta_atraso": theta,
        "req": req, "req_in": {"PO": PO, "tp": tp, "ts": ts, "erro_nulo": erro_nulo},
        "metodo": metodo, "sub_metodo": sub_metodo, "avisos": avisos, "passos": passos, "tia": tia,
    }
    n_amostras = 2 * math.pi / (req["wd"] * T)
    if n_amostras < 8:
        avisos.append(f"Poucas amostras por ciclo amortecido ({n_amostras:.1f} < 8). "
                      "T pode ser grande demais para os requisitos.")

    if metodo == "LR":
        candidatos = _candidatos_lr(R, sub_metodo, erro_nulo)
    else:
        candidatos = _candidatos_rf(R, erro_nulo, folga)
    if not candidatos:
        raise ErroProjeto("Nenhum controlador viável foi encontrado com este método. "
                          "Tente o outro método, outro T ou requisitos menos exigentes.")

    for c in candidatos:
        _avalia_candidato(R, c)
    validos = [c for c in candidatos if "falha" not in c]
    if not validos:
        raise ErroProjeto(f"Projeto inviável com este método: {candidatos[0]['falha']}. "
                          "Tente o outro método, outro T ou requisitos menos exigentes.")
    melhor = min(validos, key=lambda c: c["custo"])
    if not np.isfinite(melhor["custo"]):
        melhor = validos[0]
    classico = candidatos[0]
    if "falha" not in classico and (classico["custo"] == melhor["custo"]
                                    or (classico["estavel"] and classico["custo"] < 0.3)):
        melhor = classico
    R["candidatos"] = candidatos
    R["escolhido"] = melhor
    R["passos"].extend(melhor["passos"])
    avisos.extend(melhor.get("avisos", []))
    if melhor is not classico:
        motivo = classico.get("falha") or ("malha fechada instável" if not classico["estavel"]
                                           else "desempenho pior que o de uma variante")
        avisos.append(f"O projeto clássico ({classico['rotulo']}) foi descartado: {motivo}. "
                      f"Usada a variante: {melhor['rotulo']}.")

    for k in ("num_c", "den_c", "num_L", "den_L", "num_mf", "den_mf", "polos_mf",
              "estavel", "dc", "t", "y", "info"):
        R[k] = melhor[k]
    for k, v in melhor.get("extra", {}).items():
        R[k] = v
    if not R["estavel"]:
        avisos.append("O sistema em malha fechada é INSTÁVEL (polo com |z| >= 1). "
                      "Nenhuma variante testada estabilizou o sistema: tente o outro método, "
                      "outro T ou requisitos menos exigentes.")
    R["ess"] = 1.0 - R["dc"] if np.isfinite(R["dc"]) else np.nan

    # esforço de controle: U(z)/R(z) = T(z)/G(z) = Gc / (1 + Gc·G)
    num_u = _limpa(np.polymul(R["num_c"], R["den_z"]))
    R["num_u"], R["den_u"] = num_u, R["den_mf"]
    b = np.pad(num_u, (len(R["den_mf"]) - len(num_u), 0))
    R["u"] = signal.lfilter(b, R["den_mf"], np.ones(len(R["t"])))
    R["u_max"] = float(np.max(np.abs(R["u"]))) if R["estavel"] else np.nan
    R["u_regime"] = ganho_dc_discreto(num_u, R["den_mf"]) if R["estavel"] else np.nan

    R["zp"] = melhor["zp"]
    R["blocos"] = divide_blocos(*R["zp"])
    for bl in R["blocos"]:
        bl["tia"] = parametros_tia(bl, T)
    if tia and not representavel_matched(R["zp"]):
        avisos.append("Este Gc(z) tem raiz real em z <= 0: não há G(s) equivalente para os blocos GZ1/GZ2 "
                      "(matched). Use os coeficientes b/a em um bloco de equação de diferenças.")
    return R


def _integrador_z():
    """Integrador digital z/(z-1)."""
    return np.array([1.0, 0.0]), np.array([1.0, -1.0])


def _falha(rotulo, motivo):
    """Candidato inviável (só serve para relatar por que o clássico não foi usado)."""
    return {"rotulo": rotulo, "num_c": np.array([0.0]), "den_c": np.array([1.0]),
            "passos": [], "complexidade": 0, "falha": motivo}


def _cand_lr(rotulo, zeros, polos, num_i, den_i, Ga, zd, passos, complexidade, extra=None):
    """Monta um candidato LR: Gc(z) = Kc·Π(z-zc)/Π(z-pc)·I(z), Kc pela condição de módulo."""
    num0, den0 = num_i, den_i
    for zc in zeros:
        num0 = np.polymul(num0, [1.0, -zc])
    for pc in polos:
        den0 = np.polymul(den0, [1.0, -pc])
    num0, den0 = np.real(num0), np.real(den0)
    G0 = np.prod([zd - zc for zc in zeros]) / np.prod([zd - pc for pc in polos])
    Kc = 1.0 / abs(G0 * Ga)          # I(z) já está incluído em Ga
    passos = list(passos) + [f"Ganho pela condição de módulo: Kc = {Kc:.6g}"]
    avisos = []
    pc = polos[0]
    if abs(pc) > 1:
        avisos.append(f"O polo do controlador (|pc| = {abs(pc):.4f}) está fora do círculo unitário: "
                      "controlador instável por si só (a malha fechada pode ser estável).")
    if any(abs(z) > 1 for z in zeros):
        avisos.append("Há zero do controlador fora do círculo unitário (fase não-mínima).")
    ex = {"Kc": Kc, "zc": list(zeros), "pc": list(polos), "num_c0": num0, "den_c0": den0}
    ex.update(extra or {})
    integ = len(den_i) > 1
    zp = (Kc, list(zeros) + ([0.0] if len(num_i) > 1 else []), list(polos) + ([1.0] if integ else []))
    return {"rotulo": rotulo, "num_c": Kc * num0, "den_c": den0, "passos": passos,
            "avisos": avisos, "complexidade": complexidade, "extra": ex, "zp": zp}


def _candidatos_lr(R, sub_metodo, erro_nulo):
    T, req = R["T"], R["req"]
    num_z, den_z = R["num_z"], R["den_z"]
    passos = R["passos"]

    sd = complex(-req["sigma"], req["wd"])
    zd = np.exp(sd * T)
    R["sd"], R["zd"] = sd, zd
    passos.append(f"Polo desejado em s:  sd = {_c(sd)}")
    passos.append(f"Polo mapeado em z:   zd = e^(sd·T) = {_c(zd)}   (|zd| = {abs(zd):.4f})")

    num_i, den_i = np.array([1.0]), np.array([1.0])
    if erro_nulo:
        if _polos_integradores_z(den_z) > 0:
            passos.append("Erro nulo: a planta já possui polo em z = 1 (tipo 1). Integrador não adicionado.")
        else:
            if R.get("tia"):
                num_i, den_i = np.array([1.0]), np.array([1.0, -1.0])
                passos.append("Erro nulo: adicionado integrador 1/(z-1) ao controlador "
                              "(forma compatível com os blocos GZ: equivale a 1/s no matched).")
            else:
                num_i, den_i = _integrador_z()
                passos.append("Erro nulo: adicionado integrador z/(z-1) ao controlador.")
    R["integrador"] = len(den_i) > 1
    Ga = avalia(np.polymul(num_z, num_i), np.polymul(den_z, den_i), zd)
    passos.append(f"Fase de G(z){'·I(z)' if R['integrador'] else ''} em zd: {math.degrees(np.angle(Ga)):.2f}°")

    if sub_metodo == "cancelamento":
        return _cands_cancelamento(R, zd, Ga, num_i, den_i)
    if sub_metodo == "avanco":
        return _cands_avanco(zd, Ga, num_i, den_i)
    return _cands_bissetriz(zd, Ga, num_i, den_i)


def _cands_avanco(zd, Ga, num_i, den_i):
    """
    Avanço de fase pelo LR: zero colocado diretamente abaixo de zd (zc = Re(zd))
    e polo pela condição de ângulo. Variantes: outras posições do zero e 2-3 estágios.
    """
    cands = []
    phi = _wrap(-np.pi - np.angle(Ga))
    opcoes = [phi] + ([phi + 2 * np.pi] if phi < 0 else [])
    angs_zero = [90, 75, 105, 60, 120, 45, 135, 30]
    for i_phi, phi_tot in enumerate(opcoes):
        for n in (1, 2, 3):
            per = phi_tot / n
            for a_deg in angs_zero:
                classico = i_phi == 0 and n == 1 and a_deg == 90
                ang_zc = math.radians(a_deg)
                ang_pc = ang_zc - per
                if not (MARGEM_ANG <= ang_pc <= np.pi - MARGEM_ANG):
                    if classico:
                        cands.append(_falha("avanço de fase clássico (zero sob zd)",
                                            f"o polo precisaria de ângulo {math.degrees(ang_pc):.1f}°, "
                                            "fora de 0°–180°"))
                    continue
                zc = _angulo_para_real(zd, ang_zc)
                pc = _angulo_para_real(zd, ang_pc)
                ps = [f"Deficiência angular φ = {math.degrees(phi):.2f}°"
                      + (f"  -> compensando {math.degrees(phi_tot):.2f}°" if phi_tot != phi else "")
                      + f"   ({'avanço' if phi_tot > 0 else 'atraso'} de fase)"]
                if a_deg == 90:
                    ps.append(f"Zero colocado diretamente abaixo de zd: zc = Re(zd) = {zc:.6f}  (∠ = 90°)")
                else:
                    ps.append(f"Zero com ângulo {a_deg}° visto de zd: zc = {zc:.6f}")
                if n > 1:
                    ps.append(f"{n} estágios idênticos de {math.degrees(per):.2f}° cada")
                ps.append(f"Condição de ângulo: ∠(zd - pc) = ∠(zd - zc) - φ = {math.degrees(ang_pc):.2f}°"
                          f"  ->  pc = {pc:.6f}" + (f"  (x{n})" if n > 1 else ""))
                if classico:
                    rot = "avanço de fase clássico (zero sob zd)"
                else:
                    rot = f"avanço de fase, zero a {a_deg}°, {n} estágio(s), φ = {math.degrees(phi_tot):.1f}°"
                cands.append(_cand_lr(rot, [zc] * n, [pc] * n, num_i, den_i, Ga, zd, ps,
                                      complexidade=(n - 1) + (0 if a_deg == 90 else 0.3) + i_phi,
                                      extra={"phi": phi_tot, "estagios": n}))
    return cands


def _cands_bissetriz(zd, Ga, num_i, den_i):
    cands = []
    phi = _wrap(-np.pi - np.angle(Ga))
    theta = np.angle(zd)
    opcoes = [phi] + ([phi + 2 * np.pi] if phi < 0 else [])
    for i_phi, phi_tot in enumerate(opcoes):
        for n in (1, 2, 3):
            classico = i_phi == 0 and n == 1
            per = phi_tot / n
            lim_inf = abs(per) / 2 + MARGEM_ANG
            lim_sup = np.pi - abs(per) / 2 - MARGEM_ANG
            if lim_inf > lim_sup:
                if classico:
                    cands.append(_falha("bissetriz clássica",
                                        f"deficiência de {math.degrees(phi):.1f}° impossível com 1 polo e 1 zero"))
                continue
            centro = theta / 2
            ajustado = not (lim_inf <= centro <= lim_sup)
            if ajustado:
                if classico:
                    cands.append(_falha("bissetriz clássica",
                                        "a bissetriz pura deixaria o zero ou o polo com ângulo fora de 0°–180°"))
                centro = min(max(centro, lim_inf), lim_sup)
            ang_zc, ang_pc = centro + per / 2, centro - per / 2
            zc = _angulo_para_real(zd, ang_zc)
            pc = _angulo_para_real(zd, ang_pc)
            ps = [f"Deficiência angular φ = {math.degrees(phi):.2f}°"
                  + (f"  -> compensando {math.degrees(phi_tot):.2f}°" if phi_tot != phi else "")
                  + f"   ({'avanço' if phi_tot > 0 else 'atraso'} de fase)",
                  f"∠zd = {math.degrees(theta):.2f}°, bissetriz = {math.degrees(theta / 2):.2f}°"]
            if n > 1:
                ps.append(f"{n} estágios idênticos de {math.degrees(per):.2f}° cada")
            if ajustado:
                ps.append(f"Centro deslocado para {math.degrees(centro):.2f}° (com a bissetriz pura o zero "
                          "ou o polo ficaria com ângulo fora de 0°–180°)")
            ps.append(f"Ângulos: zero {math.degrees(ang_zc):.2f}°, polo {math.degrees(ang_pc):.2f}°")
            ps.append(f"Zero zc = {zc:.6f},  polo pc = {pc:.6f}" + (f"  (x{n})" if n > 1 else ""))
            if classico and not ajustado:
                rot = "bissetriz clássica"
            else:
                rot = (f"bissetriz{' deslocada' if ajustado else ''}, {n} estágio(s), "
                       f"φ = {math.degrees(phi_tot):.1f}°")
            cands.append(_cand_lr(rot, [zc] * n, [pc] * n, num_i, den_i, Ga, zd, ps,
                                  complexidade=(n - 1) + (0.5 if ajustado else 0) + i_phi,
                                  extra={"phi": phi_tot, "estagios": n}))
    return cands


def _grupos_polos_lentos(den_z):
    """Grupos (real ou par complexo) de polos estáveis da planta, do mais lento ao mais rápido."""
    polos = np.roots(den_z) if len(den_z) > 1 else np.array([])
    grupos, usados = [], np.zeros(len(polos), bool)
    for i, p in enumerate(polos):
        if usados[i] or abs(p - 1) < TOL_INTEGRADOR_Z or abs(p) >= 1 - 1e-9 or abs(p) < 1e-9:
            continue
        usados[i] = True
        if abs(p.imag) > 1e-9:
            j = next((j for j in range(len(polos)) if not usados[j]
                      and abs(polos[j] - np.conj(p)) < 1e-6), None)
            if j is not None:
                usados[j] = True
            grupos.append([p, np.conj(p)])
        else:
            grupos.append([p.real])
    grupos.sort(key=lambda g: -abs(g[0]))
    return grupos


def _cands_cancelamento(R, zd, Ga, num_i, den_i):
    T = R["T"]
    cands = []
    grupos = _grupos_polos_lentos(R["den_z"])
    if not grupos:
        raise ErroProjeto("A planta não tem polo estável (fora de z = 1) para cancelar. Use a Bissetriz.")
    for g in (1, 2):
        if g > len(grupos):
            break
        zeros = [z for gr in grupos[:g] for z in gr]
        m = len(zeros)
        if m > 3:
            break
        alvo = (sum(np.angle(zd - z) for z in zeros) + np.angle(Ga) + np.pi) % (2 * np.pi)
        desc = ", ".join(_c(z) if abs(np.imag(z)) > 1e-9 else f"{np.real(z):.6f}" for z in zeros)
        ps_s = [np.log(complex(z)) / T for z in zeros]
        desc_s = ", ".join(_c(p) if abs(p.imag) > 1e-9 else f"{p.real:.4f}" for p in ps_s)
        for k in range(m):
            classico = g == 1 and k == 0
            ang_pc = (alvo + 2 * np.pi * k) / m
            if not (1e-3 < ang_pc < np.pi - 1e-3):
                if classico:
                    cands.append(_falha("cancelamento do polo mais lento",
                                        f"seria necessário ∠(zd - pc) = {math.degrees(_wrap(ang_pc)):.1f}°, "
                                        "fora de 0°–180°"))
                continue
            pc = _angulo_para_real(zd, ang_pc)
            ps = [f"Polo(s) cancelado(s): z = {desc}   (s = {desc_s})  -> zero(s) do controlador",
                  f"Condição de ângulo: {m if m > 1 else ''}∠(zd - pc) = ∑∠(zd - zc) + ∠G(zd) + 180°"
                  f"  ->  ∠(zd - pc) = {math.degrees(ang_pc):.2f}°",
                  f"Polo do controlador pc = {pc:.6f}" + (f"  (x{m})" if m > 1 else "")]
            rot = ("cancelamento do polo mais lento" if m == 1
                   else f"cancelamento dos {m} polos mais lentos")
            if m > 1 and k > 0:
                rot += f" (solução {k + 1})"
            cands.append(_cand_lr(rot, zeros, [pc] * m, num_i, den_i, Ga, zd, ps,
                                  complexidade=(m - 1), extra={"cancelados": zeros}))
    return cands


def _candidatos_rf(R, erro_nulo, folga):
    T, req = R["T"], R["req"]
    num_s, den_s = R["num_s"], R["den_s"]
    passos = R["passos"]
    z = req["zeta"]

    raiz = math.sqrt(math.sqrt(1 + 4 * z ** 4) - 2 * z ** 2)
    MF_alvo = math.degrees(math.atan(2 * z / raiz))
    wc = req["wn"] * raiz
    R["MF_alvo"], R["wc_alvo"] = MF_alvo, wc
    passos.append(f"Margem de fase alvo (2ª ordem, ζ = {z:.4f}): MF = {MF_alvo:.2f}°")
    passos.append(f"Frequência de cruzamento alvo: wc = wn·√(√(1+4ζ⁴) - 2ζ²) = {wc:.4f} rad/s")
    Td = R["Td"]
    atraso = T / 2 + Td
    passos.append(f"Atraso equivalente do ZOH (T/2 = {T / 2:.4g} s): {math.degrees(wc * T / 2):.2f}° em wc")
    if Td > 0:
        passos.append(f"Atraso de transporte (Td = {Td:.4g} s): {math.degrees(wc * Td):.2f}° em wc")
    if wc * T > 0.5 * math.pi:
        R["avisos"].append("wc está próximo da frequência de Nyquist: a aproximação contínua perde precisão.")
    R["margens_orig"] = margens(num_s, den_s, Td)

    tipo1 = _polos_integradores_s(den_s) > 0
    pis = []
    if not tipo1:
        pis = [(wc / 10, "zero em wc/10"), (wc / 4, "zero em wc/4")]
        polos = np.roots(den_s) if len(den_s) > 1 else np.array([])
        reais = [-p.real for p in polos if abs(p.imag) < 1e-9 and p.real < -TOL_INTEGRADOR_S]
        if reais:
            pis.append((min(reais), "zero cancelando o polo mais lento da planta"))
    if erro_nulo:
        if tipo1:
            passos.append("Erro nulo: a planta já possui integrador (tipo 1). PI não adicionado.")
        cands = [_cand_rf(R, opc, wc, MF_alvo, folga, atraso) for opc in (pis or [None])]
    else:
        # sem exigência de erro nulo: o clássico é sem PI; variantes com PI só
        # são escolhidas se o clássico for instável ou muito pior
        cands = [_cand_rf(R, None, wc, MF_alvo, folga, atraso)]
        for opc in pis:
            c = _cand_rf(R, opc, wc, MF_alvo, folga, atraso)
            c["complexidade"] = c.get("complexidade", 0) + 4
            cands.append(c)
    return cands


def _cand_rf(R, opc_pi, wc, MF_alvo, folga, atraso):
    T = R["T"]
    num_s, den_s = R["num_s"], R["den_s"]
    ps, av = [], []
    num_c, den_c = np.array([1.0]), np.array([1.0])
    zeros_s, polos_s = [], []
    if opc_pi is not None:
        wi, desc = opc_pi
        num_c, den_c = np.array([1.0, wi]), np.array([1.0, 0.0])
        zeros_s, polos_s = [-wi], [0.0]
        ps.append(f"Erro nulo: PI (s + {wi:.4g})/s  ({desc})")

    fase = fase_continua(np.polymul(num_s, num_c), np.polymul(den_s, den_c), np.array([wc]), atraso)[0]
    fase_g = math.degrees(fase)
    phi_m = -180.0 + MF_alvo + folga - fase_g
    ps.append(f"Fase em wc (planta{' + PI' if opc_pi else ''}{' + atraso' if R['Td'] > 0 else ''} + ZOH): "
              f"{fase_g:.2f}°")
    ps.append(f"Avanço necessário: φm = -180 + {MF_alvo:.2f} + folga {folga:.1f} - ({fase_g:.2f}) = {phi_m:.2f}°")

    rot = "avanço de fase"
    if opc_pi is not None:
        rot += f" + PI ({opc_pi[1]})"
    extra = {"estagios": 0, "alpha": None}
    if phi_m > 0:
        n = int(math.ceil(phi_m / 60.0))
        if n > 3:
            return _falha(rot, f"avanço necessário de {phi_m:.1f}° (> 180°)")
        phi_e = math.radians(phi_m / n)
        alpha = (1 - math.sin(phi_e)) / (1 + math.sin(phi_e))
        Tl = 1.0 / (wc * math.sqrt(alpha))
        for _ in range(n):
            num_c = np.polymul(num_c, [Tl, 1.0])
            den_c = np.polymul(den_c, [alpha * Tl, 1.0])
        zeros_s += [-1.0 / Tl] * n
        polos_s += [-1.0 / (alpha * Tl)] * n
        extra.update({"estagios": n, "alpha": alpha, "Tl": Tl})
        if n > 1:
            av.append(f"Avanço > 60°: usados {n} estágios de avanço em cascata ({phi_m / n:.1f}° cada).")
        ps.append(f"Compensador de avanço: α = {alpha:.5f}, T = {Tl:.5g} s  "
                  f"(zero em s = {-1 / Tl:.4g}, polo em s = {-1 / (alpha * Tl):.4g})"
                  + (f"  x{n}" if n > 1 else ""))
    else:
        rot = rot.replace("avanço de fase", "ganho proporcional")
        ps.append("Avanço não necessário: apenas ganho (a MF obtida fica ≥ alvo).")

    K = 1.0 / abs(avalia(np.polymul(num_s, num_c), np.polymul(den_s, den_c), 1j * wc))
    ps.append(f"Ganho K para |C(jwc)·G(jwc)| = 1: K = {K:.6g}")
    num_cs, den_cs = _limpa(K * num_c), _limpa(den_c)
    k = den_cs[0]
    num_cs, den_cs = num_cs / k, den_cs / k

    numLs, denLs = np.polymul(num_cs, num_s), np.polymul(den_cs, den_s)
    extra.update({"num_cs": num_cs, "den_cs": den_cs, "K_rf": K, "num_Ls": numLs, "den_Ls": denLs,
                  "margens_s": margens(numLs, denLs, R["Td"]), "margens_s_zoh": margens(numLs, denLs, atraso)})

    if len(den_cs) == 1:
        num_cz, den_cz = num_cs / den_cs[0], np.array([1.0])
        zp = (float(num_cz[0]), [], [])
        ps.append("Controlador puramente proporcional: Gc(z) = K.")
    elif R.get("tia"):
        zp = matched(num_cs[0] / den_cs[0], zeros_s, polos_s, T)
        num_cz = zp[0] * np.real(np.poly(zp[1])) if zp[1] else np.array([zp[0]])
        den_cz = np.real(np.poly(zp[2]))
        ps.append("Controlador discretizado por polo-zero matched (o mesmo c2d 'matched' que os blocos "
                  "GZ do TIA fazem): cada raiz r de C(s) vai para z = e^(rT).")
    else:
        num_cz, den_cz = c2d(num_cs, den_cs, T, "tustin")
        tustin = lambda r: (1 + r * T / 2) / (1 - r * T / 2)
        zz = [tustin(r) for r in zeros_s] + [-1.0] * (len(polos_s) - len(zeros_s))
        zp = (num_cz[0] / den_cz[0], zz, [tustin(r) for r in polos_s])
        ps.append("Controlador discretizado pelo método de Tustin (bilinear): "
                  "cada raiz r de C(s) vai para z = (1 + rT/2)/(1 - rT/2).")

    # resposta contínua de referência (sem ZOH; atraso por Padé)
    if R["Td"] > 0:
        pn, pd = pade(R["Td"])
        nmf, dmf = malha_fechada(np.polymul(numLs, pn), np.polymul(denLs, pd))
    else:
        nmf, dmf = malha_fechada(numLs, denLs)
    polos = np.roots(dmf) if len(dmf) > 1 else np.array([])
    if polos.size and np.all(polos.real < 0):
        req = R["req"]
        tf_ = max(8.0 / max(min(-polos.real), 1e-3), 3 * req["ts"])
        tf_ = min(tf_, 200 * req["ts"])
        try:
            tc, yc = resposta_degrau_continua(nmf, dmf, tf_)
            extra.update({"t_cont": tc, "y_cont": yc,
                          "info_cont": stepinfo(tc, yc, ganho_dc_continuo(nmf, dmf))})
        except Exception:
            pass

    return {"rotulo": rot, "num_c": num_cz, "den_c": den_cz, "passos": ps, "avisos": av,
            "complexidade": max(0, extra["estagios"] - 1), "extra": extra, "zp": zp}


# =========================================================================
# Formatação
# =========================================================================
def _n(x):
    return f"{x:.6g}"


def _c(x):
    x = complex(x)
    sinal = "+" if x.imag >= 0 else "-"
    return f"{x.real:.6g} {sinal} j{abs(x.imag):.6g}"


def poly_str(p, var="z"):
    p = _limpa(p)
    n = len(p) - 1
    termos = []
    for i, c in enumerate(p):
        e = n - i
        if abs(c) < 1e-15 * max(1.0, np.max(np.abs(p))):
            continue
        mag = abs(c)
        if e == 0:
            corpo = _n(mag)
        else:
            coef = "" if abs(mag - 1) < 1e-12 else _n(mag) + " "
            corpo = coef + (var if e == 1 else f"{var}^{e}")
        sinal = "-" if c < 0 else "+"
        termos.append((sinal, corpo))
    if not termos:
        return "0"
    s = ("-" if termos[0][0] == "-" else "") + termos[0][1]
    for sinal, corpo in termos[1:]:
        s += f" {sinal} {corpo}"
    return s


def fatores_str(raizes, var="z"):
    """Lista de fatores (var - r), com pares complexos como quadráticos."""
    fat = []
    usados = np.zeros(len(raizes), bool)
    for i, r in enumerate(raizes):
        if usados[i]:
            continue
        usados[i] = True
        if abs(r.imag) > 1e-9:
            for j in range(i + 1, len(raizes)):
                if not usados[j] and abs(raizes[j] - np.conj(r)) < 1e-6 * max(1, abs(r)):
                    usados[j] = True
                    break
            fat.append(f"({var}^2 {'-' if r.real > 0 else '+'} {_n(abs(2 * r.real))} {var} + {_n(abs(r) ** 2)})")
        else:
            a = r.real
            if abs(a) < 1e-12:
                fat.append(var)
            else:
                fat.append(f"({var} {'-' if a > 0 else '+'} {_n(abs(a))})")
    return fat


def zpk_str(num, den, var="z", zeros=None, polos=None):
    num, den = _limpa(num), _limpa(den)
    k = num[0] / den[0]
    zs = np.roots(num) if zeros is None and len(num) > 1 else np.array(zeros if zeros is not None else [], complex)
    ps = np.roots(den) if polos is None and len(den) > 1 else np.array(polos if polos is not None else [], complex)
    s = _n(k)
    fz, fp = fatores_str(zs, var), fatores_str(ps, var)
    if fz:
        s += " · " + " ".join(fz)
    if fp:
        s += " / " + ("[" + " ".join(fp) + "]" if len(fp) > 1 else fp[0])
    return s


def eq_diferencas(num, den, saida="u", entrada="e"):
    """saida[k] = ... a partir de G(z) = Saida(z)/Entrada(z)."""
    num, den = _limpa(num), _limpa(den)
    n = len(den) - 1
    b = np.pad(num, (len(den) - len(num), 0)) / den[0]
    a = den / den[0]
    termos = []
    for i in range(1, n + 1):
        if abs(a[i]) > 1e-15:
            termos.append((-a[i], f"{saida}[k-{i}]"))
    for i in range(0, n + 1):
        if abs(b[i]) > 1e-15:
            termos.append((b[i], f"{entrada}[k]" if i == 0 else f"{entrada}[k-{i}]"))
    s = f"{saida}[k] = "
    if not termos:
        return s + "0"
    for j, (c, v) in enumerate(termos):
        if j == 0:
            s += f"{'-' if c < 0 else ''}{_n(abs(c))}·{v}"
        else:
            s += f" {'-' if c < 0 else '+'} {_n(abs(c))}·{v}"
    return s


def raizes_str(raizes, nome):
    """Lista cada raiz: 'z1 = 0.99', 'z2 = 0.5 + j0.1', ..."""
    rs = sorted(np.atleast_1d(raizes), key=lambda r: (-r.real, -r.imag))
    if not rs:
        return "nenhum"
    return ";  ".join(f"{nome}{i + 1} = " + (_c(r) if abs(r.imag) > 1e-9 else _n(r.real))
                      for i, r in enumerate(rs))


def _grupos_raizes(raizes):
    """Agrupa raízes: pares complexos conjugados juntos, reais sozinhas."""
    raizes = [complex(r) for r in raizes]
    gs, usados = [], [False] * len(raizes)
    for i, r in enumerate(raizes):
        if usados[i]:
            continue
        usados[i] = True
        if abs(r.imag) > 1e-9:
            j = next((j for j in range(len(raizes)) if not usados[j]
                      and abs(raizes[j] - r.conjugate()) < 1e-6 * max(1, abs(r))), None)
            if j is not None:
                usados[j] = True
            gs.append([r, r.conjugate()])
        else:
            gs.append([r.real])
    return gs


def _bloco(k, zeros, polos):
    num = k * (np.real(np.poly(zeros)) if zeros else np.array([1.0]))
    den = np.real(np.poly(polos)) if polos else np.array([1.0])
    return {"num": num, "den": den, "zeros": list(zeros), "polos": list(polos), "K": k}


def divide_blocos(K, zeros, polos):
    """
    Divide Gc(z) = K·Π(z - zi)/Π(z - pi) em blocos em cascata de no máximo
    2ª ordem (cada um próprio). Pares complexos ficam juntos; o ganho vai todo
    para o primeiro bloco. Retorna lista de dicts {num, den, zeros, polos, K}.
    """
    if len(polos) <= 2:
        return [_bloco(K, zeros, polos)]
    gp = _grupos_raizes(polos)
    gz = _grupos_raizes(zeros)
    # blocos de polos: pares complexos sozinhos, reais de dois em dois (mais lentos juntos)
    blocos = [{"p": g, "z": []} for g in gp if len(g) == 2]
    reais = sorted([g[0] for g in gp if len(g) == 1], key=lambda x: -abs(x))
    for i in range(0, len(reais), 2):
        blocos.append({"p": reais[i:i + 2], "z": []})
    # zeros: pares complexos primeiro (precisam de bloco com 2 polos livres)
    for g in sorted(gz, key=len, reverse=True):
        alvo = next((b for b in blocos if len(b["p"]) - len(b["z"]) >= len(g)), None)
        if alvo is None:
            raise ErroProjeto("Não foi possível dividir Gc(z) em blocos de 2ª ordem próprios.")
        alvo["z"].extend(g)
    return [_bloco(K if i == 0 else 1.0, b["z"], b["p"]) for i, b in enumerate(blocos)]


# ---------------------------------------------------------------- matched / TIA
TOL_RAIZ = 1e-9


def representavel_matched(zp):
    """True se nenhum polo/zero é real em z <= 0 (ln(z) real não existe)."""
    _, zs, ps = zp
    return all(abs(complex(r).imag) > TOL_RAIZ or complex(r).real > TOL_RAIZ for r in list(zs) + list(ps))


def _ganho_baixa_freq(k, zeros, polos, origem, avalia_fator):
    """Ganho dos fatores que não estão na 'origem' (s = 0 ou z = 1)."""
    g = k
    for r in zeros:
        if abs(r - origem) > TOL_INTEGRADOR_Z:
            g *= avalia_fator(r)
    for r in polos:
        if abs(r - origem) > TOL_INTEGRADOR_Z:
            g /= avalia_fator(r)
    return g


def _n_origem(raizes, origem):
    return sum(1 for r in raizes if abs(r - origem) <= TOL_INTEGRADOR_Z)


def matched(k_s, zeros_s, polos_s, Ts):
    """
    c2d 'matched' de G(s) = k_s·Π(s - zi)/Π(s - pi): z = e^(r·Ts) para cada raiz e ganho
    igualado em baixa frequência (ganho DC; com integradores, iguala a assíntota
    usando s ≈ (z - 1)/Ts). Zeros no infinito não são mapeados (como no MATLAB).
    Retorna (K, zeros_z, polos_z).
    """
    zz = [complex(np.exp(r * Ts)) for r in zeros_s]
    pz = [complex(np.exp(r * Ts)) for r in polos_s]
    n_int = _n_origem(polos_s, 0) - _n_origem(zeros_s, 0)
    gs = _ganho_baixa_freq(k_s, zeros_s, polos_s, 0, lambda r: -r)
    gz = _ganho_baixa_freq(1.0, zz, pz, 1, lambda r: 1 - r)
    K = float(np.real(gs / gz * Ts ** n_int))
    real = lambda lst: [r.real if abs(r.imag) < TOL_RAIZ else r for r in lst]
    return K, real(zz), real(pz)


def matched_inverso(K, zeros_z, polos_z, Ts):
    """G(s) cujo c2d 'matched' é K·Π(z - zi)/Π(z - pi). Retorna (k_s, zeros_s, polos_s)."""
    zs = [complex(np.log(complex(r))) / Ts for r in zeros_z]
    ps = [complex(np.log(complex(r))) / Ts for r in polos_z]
    n_int = _n_origem(polos_z, 1) - _n_origem(zeros_z, 1)
    gz = _ganho_baixa_freq(K, zeros_z, polos_z, 1, lambda r: 1 - r)
    gs = _ganho_baixa_freq(1.0, zs, ps, 0, lambda r: -r)
    k_s = float(np.real(gz / gs / Ts ** n_int))
    real = lambda lst: [r.real if abs(r.imag) < TOL_RAIZ else r for r in lst]
    return k_s, real(zs), real(ps)


def parametros_tia(bloco, Ts):
    """
    Parâmetros do bloco GZ1 (1ª ordem) ou GZ2 (2ª ordem) do TIA:
      GZ1: G(s) = (c0·s + c1)/(s + d1)
      GZ2: G(s) = (c0·s² + c1·s + c2)/(s² + d1·s + d2)
    Retorna dict com tipo, c, d, a G(s) e os b/a que o bloco deve calcular (conferência).
    """
    ordem = len(bloco["polos"])
    if ordem == 0:
        return {"tipo": "ganho", "K": bloco["K"]}
    zp = (bloco["K"], bloco["zeros"], bloco["polos"])
    if not representavel_matched(zp):
        return {"tipo": f"GZ{ordem}", "ok": False,
                "motivo": "raiz real em z <= 0 (sem equivalente contínuo no matched)"}
    k_s, zs, ps = matched_inverso(*zp, Ts)
    num_s = k_s * np.real(np.poly(zs)) if zs else np.array([k_s])
    den_s = np.real(np.poly(ps))
    c = np.pad(num_s, (ordem + 1 - len(num_s), 0))
    d = den_s[1:]
    # conferência: matched direto de volta
    K2, z2, p2 = matched(k_s, zs, ps, Ts)
    nz = K2 * np.real(np.poly(z2)) if z2 else np.array([K2])
    dz = np.real(np.poly(p2))
    b, a = coef_z_inv(nz, dz)
    return {"tipo": f"GZ{ordem}", "ok": True, "c": c, "d": d, "num_s": num_s, "den_s": den_s,
            "b": b, "a": a, "integrador": _n_origem(ps, 0) > 0}


def coef_z_inv(num, den):
    """Coeficientes na forma (b0 + b1 z^-1 + b2 z^-2)/(1 + a1 z^-1 + a2 z^-2)."""
    num, den = _limpa(num), _limpa(den)
    b = np.pad(num, (len(den) - len(num), 0)) / den[0]
    a = den / den[0]
    return b, a


def lugar_raizes(num_L, den_L, K_proj, n=700):
    """Pontos do lugar das raízes de 1 + k·L(z), k de 0 a ~100·K_proj."""
    num_L, den_L = _limpa(num_L), _limpa(den_L)
    ks = np.concatenate([[0.0], np.logspace(math.log10(K_proj) - 5, math.log10(K_proj) + 2, n)])
    nlen = len(den_L)
    nP = np.pad(num_L, (nlen - len(num_L), 0))
    pts = []
    for k in ks:
        r = np.roots(den_L + k * nP)
        pts.append(r)
    return np.concatenate(pts) if pts else np.array([])
