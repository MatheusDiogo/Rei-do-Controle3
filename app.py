"""
Rei do Controle - Projeto de Controlador Digital Gc(z)

Interface gráfica (tkinter + matplotlib) sobre o núcleo de cálculo em controle.py.
"""
import math
import tkinter as tk
import os
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import controle as ctl

TITULO = "Rei do Controle - Projeto de Controlador Digital"
FONTE_MONO = ("Consolas", 10)
EXEMPLO_G = "2.5537e+06 / (180.53*s^3 + 26.873*s^2 + s)"


def _num(txt, nome, obrigatorio=False):
    """Lê um número de um campo (aceita vírgula). Campo vazio -> None."""
    t = txt.strip().replace(",", ".")
    if not t:
        if obrigatorio:
            raise ctl.ErroProjeto(f"Preencha o campo '{nome}'.")
        return None
    try:
        v = float(t)
    except ValueError:
        raise ctl.ErroProjeto(f"Valor inválido em '{nome}': '{txt}'.")
    if not math.isfinite(v):
        raise ctl.ErroProjeto(f"Valor inválido em '{nome}'.")
    return v


def _fmt(v, casas=4):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{casas}f}"


def planta_str(num, den, Td):
    """G(s) como fração, com o atraso de transporte quando houver."""
    txt = fracao("G(s)", num, den, "s")
    if Td > 0:
        linhas = txt.split("\n")
        meio = len(linhas) // 2
        linhas[meio] += f" · e^(-{Td:g}·s)"
        txt = "\n".join(linhas)
    return txt


def fracao(nome, num, den, var="z"):
    """Monta a FT como fração em texto (numerador / traço / denominador)."""
    n = ctl.poly_str(num, var)
    d = ctl.poly_str(den, var)
    if d == "1":
        return f"{nome} = {n}"
    larg = max(len(n), len(d))
    pre = f"{nome} = "
    esp = " " * len(pre)
    return (f"{esp}{n.center(larg)}\n"
            f"{pre}{'-' * larg}\n"
            f"{esp}{d.center(larg)}")


NOMES_LR = {"cancelamento": "Cancelamento", "bissetriz": "Bissetriz", "avanco": "Avanço de fase"}


def _bloco_txt(nome, num, den, zs, ps, saida, entrada):
    """Fração, ganho/zeros/polos, eq. de diferenças e coeficientes em z^-1 de um bloco."""
    num, den = ctl._limpa(num), ctl._limpa(den)
    b, a = ctl.coef_z_inv(num, den)
    L = [fracao(nome, num, den), "",
         f"  Ganho K = {num[0] / den[0]:.6g}",
         "  Zeros: " + ctl.raizes_str(zs, "z"),
         "  Polos: " + ctl.raizes_str(ps, "p"),
         "  Fatorado: " + ctl.zpk_str(num, den, zeros=zs, polos=ps),
         "  Eq. diferenças: " + ctl.eq_diferencas(num, den, saida, entrada),
         "  Coeficientes em z^-1:  b = [" + ", ".join(f"{x:.6g}" for x in b) + "]   a = ["
         + ", ".join(f"{x:.6g}" for x in a) + "]"]
    return "\n".join(L)


def _tia_txt(bl, i, n_blocos, Ts):
    """Parâmetros para preencher no bloco GZ1/GZ2 do TIA."""
    p = bl["tia"]
    ent = "e" if i == 0 else f"x{i}"
    sai = "u" if i == n_blocos - 1 else f"x{i + 1}"
    cab = f"Bloco {i + 1}" + (f"  (entrada {ent} -> saída {sai})" if n_blocos > 1 else "  (entrada e -> saída u)")
    if p["tipo"] == "ganho":
        return [f"{cab}:  ganho puro, u = {p['K']:.6g}·e  (não precisa de bloco GZ)"]
    if not p["ok"]:
        return [f"{cab}:  {p['tipo']} NÃO aplicável - {p['motivo']}.",
                "    Implemente este bloco direto pela equação de diferenças (coeficientes b/a acima)."]
    f = lambda x: f"{x:.8g}"
    L = [f"{cab}:  usar {p['tipo']}"]
    if p["tipo"] == "GZ1":
        L += ["    G(s) = (c0·s + c1)/(s + d1)",
              f"    c0 = {f(p['c'][0])}", f"    c1 = {f(p['c'][1])}", f"    d1 = {f(p['d'][0])}",
              f"    Ts = {f(Ts)} s  ({Ts * 1000:g} ms)",
              f"    Conferência (o bloco deve calcular): b0 = {f(p['b'][0])}, b1 = {f(p['b'][1])}, "
              f"a1 = {f(p['a'][1])}"]
    else:
        L += ["    G(s) = (c0·s² + c1·s + c2)/(s² + d1·s + d2)",
              f"    c0 = {f(p['c'][0])}", f"    c1 = {f(p['c'][1])}", f"    c2 = {f(p['c'][2])}",
              f"    d1 = {f(p['d'][0])}", f"    d2 = {f(p['d'][1])}",
              f"    Ts = {f(Ts)} s  ({Ts * 1000:g} ms)",
              f"    Conferência (o bloco deve calcular): b0 = {f(p['b'][0])}, b1 = {f(p['b'][1])}, "
              f"b2 = {f(p['b'][2])}, a1 = {f(p['a'][1])}, a2 = {f(p['a'][2])}"]
    if p["integrador"]:
        L += ["    Obs.: bloco com integrador (d = 0): o ganho do matched foi igualado pela assíntota de baixa",
              "    frequência (s ≈ (z-1)/Ts). Confira b/a calculados pelo bloco com os valores acima."]
    return L


def texto_tia(R):
    """Seção com os valores para preencher nos blocos GZ1/GZ2 do TIA."""
    blocos = R["blocos"]
    L = ["=" * 70, "PARÂMETROS PARA O TIA (blocos GZ1/GZ2 - c2d 'matched')", "=" * 70]
    if len(blocos) > 1:
        L += ["Ligar em cascata: " + " -> ".join(
            ["e"] + [f"Bloco {i + 1}" for i in range(len(blocos))] + ["u"])]
    for i, bl in enumerate(blocos):
        L += [""] + _tia_txt(bl, i, len(blocos), R["T"])
    return "\n".join(L)


def texto_controlador(R):
    """Texto completo do controlador (forma direta + blocos em cascata, se necessário)."""
    _, zs, ps = R["zp"]
    L = [_bloco_txt("Gc(z)", R["num_c"], R["den_c"], zs, ps, "u", "e"),
         f"  T = {R['T'] * 1000:g} ms     ·     Variante usada: {R['escolhido']['rotulo']}"]
    blocos = R["blocos"]
    if len(blocos) > 1:
        cadeia = " -> ".join(["e"] + [f"Gc{i + 1}" + (f" -> x{i + 1}" if i < len(blocos) - 1 else "")
                                      for i in range(len(blocos))] + ["u"])
        L += ["", "=" * 70,
              f"IMPLEMENTAÇÃO EM CASCATA ({len(blocos)} blocos de até 2ª ordem):  "
              + "Gc(z) = " + " · ".join(f"Gc{i + 1}(z)" for i in range(len(blocos))),
              f"  Ligação: {cadeia}", "=" * 70]
        for i, bl in enumerate(blocos):
            ent = "e" if i == 0 else f"x{i}"
            sai = "u" if i == len(blocos) - 1 else f"x{i + 1}"
            L += ["", f"--- Bloco {i + 1}  (entrada {ent}, saída {sai}) ---",
                  _bloco_txt(f"Gc{i + 1}(z)", bl["num"], bl["den"], bl["zeros"], bl["polos"], sai, ent)]
    return "\n".join(L)


def texto_resultado(R):
    """Parâmetros do TIA primeiro (o que vai ser digitado), depois o detalhamento de Gc(z)."""
    return texto_tia(R) + "\n\n" + texto_controlador(R)


DESCRICAO_METODO = {
    "cancelamento": "O(s) zero(s) do Gc cancela(m) o(s) polo(s) mais lento(s) da planta; o polo do Gc sai da "
                    "condição de ângulo em zd e o ganho, da condição de módulo.",
    "bissetriz": "Zero e polo do Gc posicionados simetricamente em torno da bissetriz do ângulo de zd, "
                 "compensando a defasagem φ; ganho pela condição de módulo.",
    "avanco": "Zero do Gc diretamente abaixo de zd (zc = Re(zd)); polo pela condição de ângulo; "
              "ganho pela condição de módulo.",
    "RF": "Compensador de avanço de fase projetado no Bode para a MF e a ωc alvo (a partir de ζ e ωn) "
          "e discretizado.",
    "SINT": "Impõe-se a malha fechada T(z) desejada e calcula-se diretamente Gc = T / [G·(1 - T)].",
    "OTIM": "Topologia do Gc fixada; coeficientes escolhidos por minimização de Σ(yd - y)² "
            "(evolução diferencial), com restrições de estabilidade e de |u|.",
}


def _lista_raizes(raizes, Td=0.0):
    """Raízes em texto, agrupando as que estão em z = 0."""
    rs = list(np.atleast_1d(raizes))
    n0 = sum(1 for r in rs if abs(r) < 1e-9)
    itens = [ctl._c(r) if abs(np.imag(r)) > 1e-9 else f"{np.real(r):.6g}"
             for r in sorted((r for r in rs if abs(r) >= 1e-9), key=lambda r: (-np.real(r), -np.imag(r)))]
    if n0:
        itens.append(f"0 (x{n0}{', do atraso' if Td > 0 else ''})")
    return ", ".join(itens) if itens else "nenhum"


def _margem_txt(m):
    mg = "infinita" if not np.isfinite(m["Gm_dB"]) else f"{m['Gm_dB']:.2f} dB (ω = {m['Wcg']:.4g} rad/s)"
    mf = "infinita" if not np.isfinite(m["Pm"]) else f"{m['Pm']:.2f}° (ω = {m['Wcp']:.4g} rad/s)"
    return f"MG = {mg}   ·   MF = {mf}"


def _stepinfo_txt(info, A):
    """Bloco no formato do stepinfo do MATLAB, para degrau de amplitude A."""
    campos = [("RiseTime", info["RiseTime"], 1), ("SettlingTime", info["SettlingTime"], 1),
              ("SettlingMin", info["SettlingMin"], A), ("SettlingMax", info["SettlingMax"], A),
              ("Overshoot", info["Overshoot"], 1), ("Undershoot", info["Undershoot"], 1),
              ("Peak", info["Peak"], A), ("PeakTime", info["PeakTime"], 1)]
    return [f"{n:>16}: {v * esc:.4f}" if v is not None and np.isfinite(v) else f"{n:>16}: —"
            for n, v, esc in campos]


def relatorio(R):
    """Relatório objetivo do projeto, na ordem: planta, especificações, discretização, método,
    análise angular e margens, controlador e resultados."""
    T, q, ri, an = R["T"], R["req"], R["req_in"], R["analise"]
    A = R["amp"]
    met = {"LR": "Lugar das Raízes - " + NOMES_LR.get(R["sub_metodo"], R["sub_metodo"]),
           "RF": "Resposta em Frequência - avanço de fase + " + ("matched" if R["tia"] else "Tustin"),
           "SINT": "Síntese de FT de malha fechada",
           "OTIM": "Otimização (ISE)"}[R["metodo"]]
    sep = "=" * 78
    L = [sep, "  PROJETO DE CONTROLADOR DIGITAL", f"  Método: {met}   ·   T = {T * 1000:g} ms", sep, ""]

    # 1. Planta
    L += ["1. PLANTA"]
    if R["discreto"]:
        L += [f"   Planta informada já discretizada (T = {T * 1000:g} ms) - ver item 3.", ""]
    else:
        L += [planta_str(R["num_s"], R["den_s"], R["Td"]), "",
              "   Polos: " + _lista_raizes(np.roots(R["den_s"]) if len(R["den_s"]) > 1 else []),
              "   Zeros: " + _lista_raizes(np.roots(R["num_s"]) if len(R["num_s"]) > 1 else [])]
        if R["Td"] > 0:
            L += [f"   Atraso de transporte: Td = {R['Td']:g} s"]
        L += [""]

    # 2. Especificações
    def req(v, un):
        return "—" if v is None else f"{v:g} {un}"
    L += ["2. ESPECIFICAÇÕES",
          f"   Overshoot ≤ {req(ri['PO'], '%')}   ·   tp ≤ {req(ri['tp'], 's')}   ·   ts (2%) ≤ {req(ri['ts'], 's')}",
          f"   Erro estacionário nulo: {'sim' if ri['erro_nulo'] else 'não'}   ·   degrau de {A:g} UE"
          + (f"   ·   |u| ≤ {R['u_lim']:g} V" if R["u_lim"] else ""),
          f"   ζ = {q['zeta']:.4f}   ·   ωn = {q['wn']:.4f} rad/s   ·   ωd = {q['wd']:.4f} rad/s   ·   σ = {q['sigma']:.4f}",
          f"   Polo desejado:  sd = {ctl._c(complex(-q['sigma'], q['wd']))}   ->   "
          f"zd = e^(sd·T) = {ctl._c(an['zd'])}  (|zd| = {abs(an['zd']):.4f})"]
    if R["metodo"] == "RF" and "MF_alvo" in R:
        L += [f"   Alvos em frequência: MF = {R['MF_alvo']:.2f}°   ·   ωc = {R['wc_alvo']:.4f} rad/s"]
    for n in q["notas"]:
        L += ["   " + n]
    if "r_seq" in R:
        r = R["r_seq"]
        L += [f"   Setpoint do arquivo: {len(r)} amostras ({len(r) * T:g} s), de {r.min():g} a {r.max():g} UE"
              + ("; yd(k) do arquivo" if R.get("yd_seq") is not None else "")]
    L += [""]

    # 3. Discretização
    L += ["3. DISCRETIZAÇÃO" + ("" if R["discreto"] else f" (ZOH, T = {T * 1000:g} ms)")]
    if R["Td"] > 0:
        L += [f"   Td = {R['d_atraso']}·T + {R['theta_atraso']:.6g} s  ->  z^-{R['d_atraso']}"
              + (" + atraso fracionário (transformada Z modificada)" if R["theta_atraso"] else "")]
    L += [fracao("G(z)", R["num_z"], R["den_z"]), "",
          "   Polos: " + _lista_raizes(np.roots(R["den_z"]) if len(R["den_z"]) > 1 else [], R["Td"]),
          "   Zeros: " + _lista_raizes(np.roots(R["num_z"]) if len(ctl._limpa(R["num_z"])) > 1 else []), ""]

    # 4. Método
    chave = R["sub_metodo"] if R["metodo"] == "LR" else R["metodo"]
    L += [f"4. MÉTODO: {met}", "   " + DESCRICAO_METODO.get(chave, ""),
          f"   Variante utilizada: {R['escolhido']['rotulo']}"]
    omitir = ("Polo desejado", "Polo mapeado", "Atraso de transporte Td", "Planta informada", "Setpoint do arquivo",
              "Margem de fase alvo", "Frequência de cruzamento alvo")
    L += ["   - " + p.strip() for p in R["passos"] if not p.startswith(omitir)]
    if R["metodo"] == "RF" and "num_cs" in R:
        L += ["", fracao("C(s)", R["num_cs"], R["den_cs"], "s")]
    L += [""]

    # 5. Análise angular e margens
    L += ["5. ANÁLISE ANGULAR E MARGENS",
          f"   Em zd = {ctl._c(an['zd'])}:",
          f"     ∠G(zd)      = {an['ang_G']:9.2f}°",
          f"     Defasagem φ = {an['phi']:9.2f}°   (ângulo que o controlador precisa fornecer)",
          f"     ∠Gc(zd)     = {an['ang_Gc']:9.2f}°",
          f"     ∠Gc·G(zd)   = {an['ang_L']:9.2f}°   (condição de ângulo: -180°)",
          f"     |Gc·G(zd)|  = {an['mod_L']:9.4f}    (condição de módulo: 1)",
          "   Margens (resposta em frequência discreta, z = e^(jωT)):",
          "     Sem controlador G(z):  " + _margem_txt(an["marg_G"]),
          "     Com controlador Gc·G:  " + _margem_txt(an["marg_L"])]
    if R["metodo"] in ("SINT", "OTIM"):
        L += ["   Obs.: neste método zd é a referência dos requisitos; o Gc não é calculado pela condição de ângulo."]
    if R["metodo"] == "LR" and R.get("integrador"):
        L += ["   Obs.: Gc inclui o integrador do erro nulo; a deficiência φ do item 4 é a do compensador "
              "(descontado o integrador)."]
    if R["metodo"] == "RF" and "margens_s" in R:
        ms, mz = R["margens_s"], R["margens_s_zoh"]
        L += ["   Contínuo C(s)·G(s):    " + _margem_txt(ms),
              "   C(s)·G(s) + ZOH (T/2): " + _margem_txt(mz)]
    L += [""]

    # 6. Controlador
    L += ["6. CONTROLADOR CALCULADO", texto_controlador(R), "", texto_tia(R), ""]

    # 7. Resultados
    i = R["info"]
    L += ["7. RESULTADOS (malha fechada digital)",
          f"   stepinfo({A:g}·T(z))  (degrau de {A:g} UE):" if A != 1 else "   stepinfo(T(z)):"]
    L += _stepinfo_txt(i, A)
    L += ["",
          f"   Valor de regime    : {_fmt(R['dc'] * A if np.isfinite(R['dc']) else np.nan)}   ·   "
          f"erro estacionário = {_fmt(abs(R['ess']) * 100 if np.isfinite(R['ess']) else np.nan, 4)} %",
          f"   Esforço de controle: máx |u| = {_fmt(R['u_max'] * A, 4)} V   ·   u em regime = "
          f"{_fmt(R['u_regime'] * A, 4)} V" + (f"   (limite ±{R['u_lim']:g} V)" if R["u_lim"] else ""),
          "   Polos de malha fechada: " + _lista_raizes(R["polos_mf"], R["Td"]),
          f"   Estabilidade: {'estável' if R['estavel'] else 'INSTÁVEL'} "
          f"(máx |polo| = {np.max(np.abs(R['polos_mf'])) if len(R['polos_mf']) else 0:.4f})"]
    if "y_r" in R:
        L += [f"   Setpoint do arquivo: máx |u| = {_fmt(R.get('u_max_r', np.nan), 4)} V"
              + (f"   ·   SomaErro Σ(yd - y)² = {R['soma_erro']:.6g}" if "soma_erro" in R else "")]

    def verif(nome, obtido, alvo, un):
        if alvo is None:
            return None
        if not np.isfinite(obtido):
            marca = "✗"
        else:
            marca = "✓" if obtido <= alvo * (1 + 1e-9) else ("≈" if obtido <= alvo * 1.1 else "✗")
        return f"     {nome:<22} {_fmt(obtido, 4):>10} {un:<2} (≤ {alvo:g} {un})   {marca}"
    sem_pico = np.isfinite(i["Overshoot"]) and i["Overshoot"] < 0.5
    linhas = [verif("Overshoot", i["Overshoot"], ri["PO"], "%"),
              None if sem_pico else verif("Tempo de pico", i["PeakTime"], ri["tp"], "s"),
              verif("Tempo de acomodação", i["SettlingTime"], ri["ts"], "s"),
              verif("|u| máx", (R["u_max_r"] if "u_max_r" in R else R["u_max"] * A), R["u_lim"], "V")]
    if ri["erro_nulo"]:
        ok = np.isfinite(R["ess"]) and abs(R["ess"]) < 1e-6
        linhas.append(f"     {'Erro estacionário':<22} {_fmt(abs(R['ess']) * 100, 4):>10} %  (= 0)   {'✓' if ok else '✗'}")
    linhas = [x for x in linhas if x]
    if linhas:
        L += ["   Verificação dos requisitos:"] + linhas

    if R["avisos"]:
        L += ["", "AVISOS"] + ["   • " + a for a in R["avisos"]]
    if len(R["candidatos"]) > 1:
        L += ["", "ANEXO - variantes avaliadas"]
        for c in R["candidatos"]:
            if "falha" in c:
                L += [f"   - {c['rotulo']}: inviável ({c['falha']})"]
            elif c["estavel"]:
                ci = c["info"]
                L += [f"   - {c['rotulo']}: PO = {_fmt(ci['Overshoot'], 2)} %, ts = {_fmt(ci['SettlingTime'], 3)} s"
                      + ("   <== usada" if c is R["escolhido"] else "")]
            else:
                L += [f"   - {c['rotulo']}: instável" + ("   <== usada" if c is R["escolhido"] else "")]
    return "\n".join(L)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(TITULO)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1400, sw - 40)}x{min(900, sh - 80)}+10+10")
        self.minsize(min(1000, sw - 40), min(560, sh - 80))
        if sh < 950:                      # tela baixa: abre maximizada
            try:
                self.state("zoomed")
            except tk.TclError:
                pass
        self.resultado = None
        self.series = {"r": None, "yd": None}
        self._estilo()
        self._monta_entrada()
        self._monta_saida()
        self._atualiza_metodo()

    # ------------------------------------------------------------------ UI
    def _estilo(self):
        st = ttk.Style(self)
        try:
            st.theme_use("vista")
        except tk.TclError:
            st.theme_use("clam")
        st.configure("Titulo.TLabel", font=("Segoe UI", 13, "bold"))
        st.configure("Calc.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        st.configure("Dica.TLabel", foreground="#666666", font=("Segoe UI", 8))
        st.configure("Treeview", rowheight=24, font=("Segoe UI", 10))
        st.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _monta_painel_rolavel(self):
        """Painel esquerdo: formulário com scroll vertical + rodapé fixo (botões sempre visíveis)."""
        lado = ttk.Frame(self)
        lado.pack(side=tk.LEFT, fill=tk.Y)
        self._rodape = ttk.Frame(lado, padding=(10, 4, 10, 8))
        self._rodape.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(lado, orient=tk.HORIZONTAL).pack(side=tk.BOTTOM, fill=tk.X)
        area = ttk.Frame(lado)
        area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        fundo = ttk.Style(self).lookup("TFrame", "background") or self.cget("bg")
        canvas = tk.Canvas(area, highlightthickness=0, borderwidth=0, background=fundo)
        sb = ttk.Scrollbar(area, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        esq = ttk.Frame(canvas, padding=10)
        canvas.create_window((0, 0), window=esq, anchor="nw")

        def ajusta(_=None):
            canvas.configure(scrollregion=canvas.bbox("all"), width=esq.winfo_reqwidth())
            if esq.winfo_reqheight() > canvas.winfo_height() > 1:
                if not sb.winfo_ismapped():
                    sb.pack(side=tk.RIGHT, fill=tk.Y)
            elif sb.winfo_ismapped():
                sb.pack_forget()
                canvas.yview_moveto(0)

        esq.bind("<Configure>", ajusta)
        canvas.bind("<Configure>", ajusta)

        def roda(ev):
            if sb.winfo_ismapped():
                canvas.yview_scroll(int(-ev.delta / 120) or (-1 if ev.delta > 0 else 1), "units")

        area.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", roda))
        area.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))
        return esq

    def _monta_entrada(self):
        esq = self._monta_painel_rolavel()

        ttk.Label(esq, text="Rei do Controle", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(esq, text="Projeto de controlador digital Gc(z)", style="Dica.TLabel").pack(anchor="w", pady=(0, 4))

        # Planta
        fp = ttk.LabelFrame(esq, text=" Planta ", padding=(8, 4))
        fp.pack(fill=tk.X, pady=2)
        ttk.Label(fp, text="G(s) =  (ou G(z) =)").grid(row=0, column=0, sticky="w")
        self.var_G = tk.StringVar(value=EXEMPLO_G)
        eg = ttk.Entry(fp, textvariable=self.var_G, width=42, font=FONTE_MONO)
        eg.grid(row=1, column=0, columnspan=2, sticky="we", pady=2)
        ttk.Label(fp, text="Ex: 10*(s+2)/((s+1)*(s+5))  ·  2*e^(-0.5*s)/(3*s+1)\n"
                           "Planta já discreta: use z, ex. (49.75*z + 44.97)/(z^2 - 0.852*z - 0.05611)\n"
                           "Multiplicação somente com * ; potência com ^ ; '.' = decimal.",
                  style="Dica.TLabel").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(fp, text="Verificar G(s)", command=self.verifica_planta).grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.lbl_planta = ttk.Label(fp, text="", font=("Consolas", 9), foreground="#1a5fb4", justify=tk.LEFT)
        self.lbl_planta.grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Label(fp, text="Período de amostragem T (ms):").grid(row=5, column=0, sticky="w", pady=(4, 0))
        self.var_T = tk.StringVar(value="100")
        ttk.Entry(fp, textvariable=self.var_T, width=10).grid(row=5, column=1, sticky="e", pady=(8, 0))
        fp.columnconfigure(0, weight=1)

        # Requisitos
        fr = ttk.LabelFrame(esq, text=" Requisitos (deixe em branco o que não for pedido) ", padding=(8, 4))
        fr.pack(fill=tk.X, pady=2)
        self.var_PO = tk.StringVar(value="10")
        self.var_tp = tk.StringVar(value="2")
        self.var_ts = tk.StringVar(value="")
        for i, (rot, var) in enumerate((("Overshoot máximo (%):", self.var_PO),
                                        ("Tempo de pico (s):", self.var_tp),
                                        ("Tempo de acomodação 2% (s):", self.var_ts))):
            ttk.Label(fr, text=rot).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(fr, textvariable=var, width=10).grid(row=i, column=1, sticky="e", pady=2)
        self.var_erro = tk.BooleanVar(value=True)
        ttk.Checkbutton(fr, text="Erro estacionário nulo (entrada degrau)",
                        variable=self.var_erro).grid(row=3, column=0, columnspan=2, sticky="w", pady=(3, 0))
        self.var_amp = tk.StringVar(value="1")
        self.var_ulim = tk.StringVar(value="")
        ttk.Label(fr, text="Amplitude do degrau (UE):").grid(row=4, column=0, sticky="w", pady=(6, 2))
        ttk.Entry(fr, textvariable=self.var_amp, width=10).grid(row=4, column=1, sticky="e", pady=(6, 2))
        ttk.Label(fr, text="Limite de |u| (V) (opcional):").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Entry(fr, textvariable=self.var_ulim, width=10).grid(row=5, column=1, sticky="e", pady=2)
        fr.columnconfigure(0, weight=1)

        # Sinais do TIA
        fs = ttk.LabelFrame(esq, text=" Setpoint do TIA (opcional, .txt: 1 amostra por T) ", padding=(8, 4))
        fs.pack(fill=tk.X, pady=2)
        self.lbl_serie = {}
        for i, (chave, rot) in enumerate((("r", "Setpoint r(k):"), ("yd", "Desejada yd(k):"))):
            ttk.Label(fs, text=rot).grid(row=i, column=0, sticky="w", pady=1)
            lbl = ttk.Label(fs, text="(nenhum)", style="Dica.TLabel", width=24)
            lbl.grid(row=i, column=1, sticky="w", padx=4)
            self.lbl_serie[chave] = lbl
            ttk.Button(fs, text="Abrir...", width=7,
                       command=lambda c=chave: self.carrega_serie(c)).grid(row=i, column=2, padx=1)
            ttk.Button(fs, text="x", width=2,
                       command=lambda c=chave: self.limpa_serie(c)).grid(row=i, column=3)
        ttk.Label(fs, text="Sem yd: yd = T(z) desejada aplicada ao setpoint.", style="Dica.TLabel").grid(
            row=2, column=0, columnspan=4, sticky="w")
        fs.columnconfigure(1, weight=1)

        # Método
        fm = ttk.LabelFrame(esq, text=" Método de projeto ", padding=(8, 4))
        fm.pack(fill=tk.X, pady=2)
        self.var_met = tk.StringVar(value="LR")
        ttk.Radiobutton(fm, text="Lugar das Raízes (LR)", value="LR", variable=self.var_met,
                        command=self._atualiza_metodo).grid(row=0, column=0, columnspan=2, sticky="w")
        self.var_sub = tk.StringVar(value="cancelamento")
        self.rb_canc = ttk.Radiobutton(fm, text="Cancelamento (polo mais lento)", value="cancelamento",
                                       variable=self.var_sub)
        self.rb_bis = ttk.Radiobutton(fm, text="Bissetriz", value="bissetriz", variable=self.var_sub)
        self.rb_av = ttk.Radiobutton(fm, text="Avanço de fase (zero sob zd)", value="avanco",
                                     variable=self.var_sub)
        self.rb_canc.grid(row=1, column=0, columnspan=2, sticky="w", padx=(22, 0))
        self.rb_bis.grid(row=2, column=0, columnspan=2, sticky="w", padx=(22, 0))
        self.rb_av.grid(row=3, column=0, columnspan=2, sticky="w", padx=(22, 0))
        ttk.Radiobutton(fm, text="Resposta em Frequência (RF) – avanço + Tustin", value="RF",
                        variable=self.var_met, command=self._atualiza_metodo).grid(row=4, column=0, columnspan=2,
                                                                                    sticky="w", pady=(3, 0))
        self.lbl_folga = ttk.Label(fm, text="Folga de fase (°):")
        self.lbl_folga.grid(row=5, column=0, sticky="w", padx=(22, 0))
        self.var_folga = tk.StringVar(value="5")
        self.ent_folga = ttk.Entry(fm, textvariable=self.var_folga, width=8)
        self.ent_folga.grid(row=5, column=1, sticky="e")
        ttk.Radiobutton(fm, text="Síntese de FT: Gc = T / [G·(1 - T)]", value="SINT",
                        variable=self.var_met, command=self._atualiza_metodo).grid(row=6, column=0, columnspan=2,
                                                                                    sticky="w", pady=(3, 0))
        ttk.Radiobutton(fm, text="Otimização (ISE) com limite de |u|", value="OTIM",
                        variable=self.var_met, command=self._atualiza_metodo).grid(row=7, column=0, columnspan=2,
                                                                                    sticky="w", pady=(3, 0))
        self.lbl_ordem = ttk.Label(fm, text="Ordem do Gc:")
        self.lbl_ordem.grid(row=8, column=0, sticky="w", padx=(22, 0))
        self.var_ordem = tk.StringVar(value="2")
        self.cb_ordem = ttk.Combobox(fm, textvariable=self.var_ordem, values=("1", "2"), width=5, state="readonly")
        self.cb_ordem.grid(row=8, column=1, sticky="e")
        self.var_suave = tk.BooleanVar(value=True)
        self.ck_suave = ttk.Checkbutton(fm, text="Suavizar u (sem polos do Gc em Re < 0)", variable=self.var_suave)
        self.ck_suave.grid(row=10, column=0, columnspan=2, sticky="w", padx=(22, 0))
        self.var_tia = tk.BooleanVar(value=True)
        ttk.Checkbutton(fm, text="Compatível com blocos GZ1/GZ2 do TIA (matched)",
                        variable=self.var_tia).grid(row=11, column=0, columnspan=2, sticky="w", pady=(4, 0))
        fm.columnconfigure(0, weight=1)

        # Botões
        fb = ttk.Frame(self._rodape)
        fb.pack(fill=tk.X, pady=(2, 4))
        ttk.Button(fb, text="CALCULAR", style="Calc.TButton", command=self.calcula).pack(fill=tk.X)
        fb2 = ttk.Frame(self._rodape)
        fb2.pack(fill=tk.X)
        ttk.Button(fb2, text="Copiar Gc(z)", command=self.copia_gc).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(fb2, text="Copiar relatório", command=self.copia_relatorio).pack(side=tk.LEFT, expand=True,
                                                                                    fill=tk.X, padx=4)
        ttk.Button(fb2, text="Limpar", command=self.limpa).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.lbl_status = ttk.Label(self._rodape, text="Preencha os dados e clique em CALCULAR.", wraplength=330,
                                    style="Dica.TLabel")
        self.lbl_status.pack(anchor="w", pady=(4, 0))
        self.bind("<Return>", lambda e: self.calcula())

    def _monta_saida(self):
        dir_ = ttk.Frame(self, padding=(0, 10, 10, 10))
        dir_.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.nb = ttk.Notebook(dir_)
        self.nb.pack(fill=tk.BOTH, expand=True)

        # --- Aba Resultado
        aba = ttk.Frame(self.nb, padding=8)
        self.nb.add(aba, text="  Resultado  ")
        ttk.Label(aba, text="Controlador digital", style="Titulo.TLabel").pack(anchor="w")
        fgc = ttk.Frame(aba)
        fgc.pack(fill=tk.BOTH, expand=True, pady=(2, 8))
        sgc = ttk.Scrollbar(fgc)
        sgc.pack(side=tk.RIGHT, fill=tk.Y)
        sgx = ttk.Scrollbar(fgc, orient=tk.HORIZONTAL)
        sgx.pack(side=tk.BOTTOM, fill=tk.X)
        self.txt_gc = tk.Text(fgc, height=14, font=("Consolas", 10), wrap=tk.NONE, relief=tk.FLAT,
                              background="#f6f8fa", padx=10, pady=8, yscrollcommand=sgc.set,
                              xscrollcommand=sgx.set)
        self.txt_gc.pack(fill=tk.BOTH, expand=True)
        sgc.config(command=self.txt_gc.yview)
        sgx.config(command=self.txt_gc.xview)
        self.txt_gc.configure(state=tk.DISABLED)

        ttk.Label(aba, text="Especificações alcançadas (malha fechada digital)", style="Titulo.TLabel").pack(anchor="w")
        cols = ("esp", "des", "alc", "ok")
        self.tab = ttk.Treeview(aba, columns=cols, show="headings", height=11)
        for c, t, w in zip(cols, ("Especificação", "Desejado", "Alcançado", "Status"), (280, 140, 140, 90)):
            self.tab.heading(c, text=t)
            self.tab.column(c, width=w, anchor="w" if c == "esp" else "center")
        self.tab.tag_configure("ok", foreground="#1b7f3b")
        self.tab.tag_configure("quase", foreground="#b8860b")
        self.tab.tag_configure("falha", foreground="#c01c28")
        self.tab.pack(fill=tk.X, pady=(2, 8))

        ttk.Label(aba, text="Avisos", style="Titulo.TLabel").pack(anchor="w")
        self.txt_av = tk.Text(aba, height=4, font=("Segoe UI", 9), wrap=tk.WORD, relief=tk.FLAT,
                              background="#fffbea", padx=8, pady=6)
        self.txt_av.pack(fill=tk.X, pady=(2, 0))
        self.txt_av.configure(state=tk.DISABLED)

        # --- Aba Gráficos
        abag = ttk.Frame(self.nb)
        self.nb.add(abag, text="  Gráficos  ")
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=abag)
        NavigationToolbar2Tk(self.canvas, abag).update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Aba Memória de cálculo
        abam = ttk.Frame(self.nb, padding=4)
        self.nb.add(abam, text="  Memória de cálculo  ")
        sb = ttk.Scrollbar(abam)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_mem = tk.Text(abam, font=FONTE_MONO, wrap=tk.NONE, yscrollcommand=sb.set, padx=8, pady=6)
        self.txt_mem.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.txt_mem.yview)

    def _atualiza_metodo(self):
        lr = self.var_met.get() == "LR"
        for w in (self.rb_canc, self.rb_bis, self.rb_av):
            w.configure(state=tk.NORMAL if lr else tk.DISABLED)
        rf = self.var_met.get() == "RF"
        otim = self.var_met.get() == "OTIM"
        self.ent_folga.configure(state=tk.NORMAL if rf else tk.DISABLED)
        self.lbl_folga.configure(state=tk.NORMAL if rf else tk.DISABLED)
        self.cb_ordem.configure(state="readonly" if otim else tk.DISABLED)
        self.lbl_ordem.configure(state=tk.NORMAL if otim else tk.DISABLED)
        self.ck_suave.configure(state=tk.NORMAL if otim else tk.DISABLED)

    @staticmethod
    def _set_text(w, txt):
        w.configure(state=tk.NORMAL)
        w.delete("1.0", tk.END)
        w.insert("1.0", txt)
        w.configure(state=tk.DISABLED)

    # ------------------------------------------------------------ ações
    def verifica_planta(self):
        try:
            num, den, Td, var = ctl.parse_planta(self.var_G.get())
            polos = np.roots(den) if len(den) > 1 else []
            txt = planta_str(num, den, Td) if var == "s" else fracao("G(z)", num, den, "z")
            txt += "\nPolos: " + ", ".join(ctl._c(p) if abs(p.imag) > 1e-12 else f"{p.real:.5g}" for p in polos)
            self.lbl_planta.configure(text=txt, foreground="#1a5fb4")
        except ctl.ErroProjeto as e:
            self.lbl_planta.configure(text=str(e), foreground="#c01c28")

    def carrega_serie(self, chave):
        caminho = filedialog.askopenfilename(
            title="Abrir setpoint r(k)" if chave == "r" else "Abrir resposta desejada yd(k)",
            filetypes=[("Texto", "*.txt *.csv *.dat"), ("Todos", "*.*")])
        if not caminho:
            return
        try:
            v = ctl.ler_serie(caminho)
        except ctl.ErroProjeto as e:
            messagebox.showerror("Arquivo inválido", str(e))
            return
        self.series[chave] = v
        nome = os.path.basename(caminho)
        self.lbl_serie[chave].configure(text=f"{nome[:14]}: {len(v)} pts [{v.min():g}..{v.max():g}]")

    def limpa_serie(self, chave):
        self.series[chave] = None
        self.lbl_serie[chave].configure(text="(nenhum)")

    def limpa(self):
        for v in (self.var_G, self.var_PO, self.var_tp, self.var_ts):
            v.set("")
        self.lbl_planta.configure(text="")
        self.resultado = None
        self._set_text(self.txt_gc, "")
        self._set_text(self.txt_av, "")
        self.txt_mem.delete("1.0", tk.END)
        self.tab.delete(*self.tab.get_children())
        self.fig.clear()
        self.canvas.draw()
        self.lbl_status.configure(text="Campos limpos.")

    def copia_gc(self):
        if not self.resultado:
            return
        self.clipboard_clear()
        self.clipboard_append(texto_resultado(self.resultado))
        self.lbl_status.configure(text="Gc(z) copiado para a área de transferência.")

    def copia_relatorio(self):
        if not self.resultado:
            return
        self.clipboard_clear()
        self.clipboard_append(self.txt_mem.get("1.0", tk.END))
        self.lbl_status.configure(text="Relatório copiado para a área de transferência.")

    def calcula(self):
        try:
            T_ms = _num(self.var_T.get(), "Período de amostragem T (ms)", obrigatorio=True)
            T = T_ms / 1000.0
            PO = _num(self.var_PO.get(), "Overshoot")
            tp = _num(self.var_tp.get(), "Tempo de pico")
            ts = _num(self.var_ts.get(), "Tempo de acomodação")
            folga = _num(self.var_folga.get(), "Folga de fase") or 0.0
            amp = _num(self.var_amp.get(), "Amplitude do degrau") or 1.0
            u_lim = _num(self.var_ulim.get(), "Limite de |u|")
            if u_lim is not None and u_lim <= 0:
                raise ctl.ErroProjeto("O limite de |u| deve ser positivo.")
            if self.var_met.get() == "OTIM":
                self.lbl_status.configure(text="Otimizando... (pode levar alguns segundos)")
            self.config(cursor="watch")
            self.update_idletasks()
            R = ctl.projeta(self.var_G.get(), T, PO, tp, ts, self.var_erro.get(),
                            self.var_met.get(), self.var_sub.get(), folga, tia=self.var_tia.get(),
                            ordem_otim=int(self.var_ordem.get()), amp=amp, u_lim=u_lim,
                            suavizar=self.var_suave.get(), r_seq=self.series["r"], yd_seq=self.series["yd"])
        except ctl.ErroProjeto as e:
            self.config(cursor="")
            messagebox.showerror("Não foi possível projetar", str(e))
            self.lbl_status.configure(text=str(e))
            return
        except Exception as e:  # erro numérico inesperado
            self.config(cursor="")
            messagebox.showerror("Erro inesperado", f"{type(e).__name__}: {e}")
            return
        self.config(cursor="")
        self.resultado = R
        self.verifica_planta()
        self._mostra_gc(R)
        self._mostra_tabela(R)
        self._mostra_avisos(R)
        self._mostra_memoria(R)
        self._mostra_graficos(R)
        self.nb.select(0)
        self.lbl_status.configure(
            text=("Projeto concluído: " if R["estavel"] else "Projeto concluído com sistema INSTÁVEL: ")
            + R["escolhido"]["rotulo"] + ".")

    # ------------------------------------------------------------ saída
    def _mostra_gc(self, R):
        self._set_text(self.txt_gc, texto_resultado(R))

    def _mostra_tabela(self, R):
        self.tab.delete(*self.tab.get_children())
        ri, info, req = R["req_in"], R["info"], R["req"]
        sem_pico = np.isfinite(info["Overshoot"]) and info["Overshoot"] < 0.5

        def status(alc, des, tol=0.10):
            if des is None or alc is None or not np.isfinite(alc):
                return ("—" if des is None else "✗", "" if des is None else "falha")
            if alc <= des * (1 + 1e-9):
                return "✓", "ok"
            if alc <= des * (1 + tol):
                return "≈", "quase"
            return "✗", "falha"

        linhas = []
        st = status(info["Overshoot"], ri["PO"])
        if ri["PO"] is not None and np.isfinite(info["Overshoot"]) and info["Overshoot"] <= 0.5 and ri["PO"] > 0:
            st = ("✓", "ok")
        linhas.append(("Overshoot (%)", _fmt(ri["PO"], 2) if ri["PO"] is not None else "—",
                       _fmt(info["Overshoot"], 2), st))
        if sem_pico:
            st = ("—", "") if ri["tp"] is None else ("sem pico", "quase")
            linhas.append(("Tempo de pico (s)", _fmt(ri["tp"]) if ri["tp"] is not None else "—", "sem pico", st))
        else:
            linhas.append(("Tempo de pico (s)", _fmt(ri["tp"]) if ri["tp"] is not None else "—",
                           _fmt(info["PeakTime"]), status(info["PeakTime"], ri["tp"])))
        linhas.append(("Tempo de acomodação 2% (s)", _fmt(ri["ts"]) if ri["ts"] is not None else "—",
                       _fmt(info["SettlingTime"]), status(info["SettlingTime"], ri["ts"])))
        linhas.append(("Tempo de subida 10–90% (s)", "—", _fmt(info["RiseTime"]), ("—", "")))
        ess = R["ess"]
        if ri["erro_nulo"]:
            ok = np.isfinite(ess) and abs(ess) < 1e-6
            st = ("✓", "ok") if ok else ("✗", "falha")
            linhas.append(("Erro estacionário (degrau)", "0", _fmt(ess, 6), st))
        else:
            linhas.append(("Erro estacionário (degrau)", "—", _fmt(ess, 6), ("—", "")))
        linhas.append(("Valor de regime", "1" if ri["erro_nulo"] else "—", _fmt(R["dc"]), ("—", "")))
        pm = np.max(np.abs(R["polos_mf"])) if len(R["polos_mf"]) else 0.0
        linhas.append(("Estabilidade (máx |polo MF|)", "< 1", _fmt(pm),
                       ("✓", "ok") if R["estavel"] else ("✗", "falha")))
        A = R["amp"]
        umax = R["u_max"] * A if np.isfinite(R["u_max"]) else np.nan
        st = ("—", "")
        if R["u_lim"]:
            st = ("✓", "ok") if np.isfinite(umax) and umax <= R["u_lim"] * (1 + 1e-9) else ("✗", "falha")
        linhas.append((f"Esforço máx |u| (V) p/ degrau de {A:g} UE",
                       f"≤ {R['u_lim']:g}" if R["u_lim"] else "—", _fmt(umax), st))
        if "u_max_r" in R:
            ur = R["u_max_r"]
            st = ("—", "")
            if R["u_lim"]:
                st = ("✓", "ok") if np.isfinite(ur) and ur <= R["u_lim"] * (1 + 1e-9) else ("✗", "falha")
            linhas.append(("Esforço máx |u| (V) no setpoint do arquivo",
                           f"≤ {R['u_lim']:g}" if R["u_lim"] else "—", _fmt(ur), st))
        if "soma_erro" in R:
            linhas.append(("SomaErro Σ(yd - y)² no setpoint", "mínimo", f"{R['soma_erro']:.6g}", ("—", "")))
        linhas.append((f"u em regime (V) p/ degrau de {A:g} UE", "—",
                       _fmt(R["u_regime"] * A if np.isfinite(R["u_regime"]) else np.nan), ("—", "")))
        linhas.append(("ζ desejado / ωn desejado (rad/s)", f"{req['zeta']:.4f} / {req['wn']:.4f}", "", ("", "")))
        if R["metodo"] == "RF" and "margens_s_zoh" in R:
            m = R["margens_s_zoh"]
            linhas.append(("Margem de fase c/ ZOH (°)", _fmt(R["MF_alvo"], 2), _fmt(m["Pm"], 2),
                           ("✓", "ok") if m["Pm"] >= R["MF_alvo"] - 1 else ("✗", "falha")))
            linhas.append(("Freq. de cruzamento ωc (rad/s)", _fmt(R["wc_alvo"]), _fmt(m["Wcp"]), ("—", "")))
        for esp, des, alc, (s, tag) in linhas:
            self.tab.insert("", tk.END, values=(esp, des, alc, s), tags=(tag,) if tag else ())

    def _mostra_avisos(self, R):
        av = R["avisos"]
        self._set_text(self.txt_av, "\n".join("• " + a for a in av) if av else "Nenhum aviso.")

    def _mostra_memoria(self, R):
        self.txt_mem.delete("1.0", tk.END)
        self.txt_mem.insert("1.0", relatorio(R))

    def _mostra_graficos(self, R):
        fig = self.fig
        fig.clear()
        tem_r = "y_r" in R
        gs = fig.add_gridspec(2, 3 if tem_r else 2)
        if tem_r:
            ax_r = fig.add_subplot(gs[0, 2])
            self._plot_setpoint(ax_r, R)
            self._plot_esforco(fig.add_subplot(gs[1, 2], sharex=ax_r), R, setpoint=True)
        if R["metodo"] != "RF":
            self._plot_lgr(fig.add_subplot(gs[:, 0]), R)
        else:
            axm = fig.add_subplot(gs[0, 0])
            self._plot_bode(axm, fig.add_subplot(gs[1, 0], sharex=axm), R)
        ax_y = fig.add_subplot(gs[0, 1])
        self._plot_degrau(ax_y, R)
        self._plot_esforco(fig.add_subplot(gs[1, 1], sharex=ax_y), R)
        fig.tight_layout()
        self.canvas.draw()

    def _plot_setpoint(self, ax, R):
        r = R["r_seq"]
        t = np.arange(len(r)) * R["T"]
        ax.step(t, r, where="post", color="#7f7f7f", lw=1.0, label="Setpoint r(k)")
        if "yd_r" in R:
            ax.step(t, R["yd_r"], where="post", color="#2ca02c", lw=1.2, ls="--", label="Desejada yd(k)")
        if R["estavel"]:
            ax.step(t, R["y_r"], where="post", color="#d62728", lw=1.4, label="y(k) com Gc(z)")
        tit = "Setpoint do arquivo"
        if "soma_erro" in R:
            tit += f" (SomaErro = {R['soma_erro']:.4g})"
        ax.set_title(tit, fontsize=9)
        ax.set_ylabel("Saída (UE)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    def _plot_esforco(self, ax, R, setpoint=False):
        A = 1.0 if setpoint else R["amp"]
        if setpoint:
            u = R["u_r"]
            t = np.arange(len(u)) * R["T"]
            rot = "u para o setpoint do arquivo"
        else:
            t, u = R["t"], R["u"] * A
            rot = f"u = T(z)/G(z) · degrau de {A:g} UE"
        ax.step(t, u, where="post", color="#9467bd", lw=1.4, label=rot)
        if R["estavel"] and np.all(np.isfinite(u)) and len(u):
            i = int(np.argmax(np.abs(u)))
            ax.plot(t[i], u[i], "o", color="#d62728", ms=5)
            ax.annotate(f"máx |u| = {abs(u[i]):.4g} V  (t = {t[i]:.3g} s)", (t[i], u[i]),
                        textcoords="offset points", xytext=(8, -12 if u[i] > 0 else 8), fontsize=8,
                        color="#d62728")
            if np.isfinite(R["u_regime"]) and not setpoint:
                ax.axhline(R["u_regime"] * A, color="k", lw=0.7, ls="--",
                           label=f"u em regime = {R['u_regime'] * A:.4g} V")
        if R["u_lim"]:
            for s in (1, -1):
                ax.axhline(s * R["u_lim"], color="#c01c28", lw=1.0, ls=":",
                           label=f"Limite ±{R['u_lim']:g} V" if s == 1 else None)
        ax.set_title("Esforço u" + (" (setpoint)" if setpoint else " (degrau)"), fontsize=9)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("u (V)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    def _plot_lgr(self, ax, R):
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8, label="Círculo unitário")
        numL = np.polymul(R["num_c0"], R["num_z"])
        denL = np.polymul(R["den_c0"], R["den_z"])
        pts = ctl.lugar_raizes(numL, denL, R["Kc"])
        ax.plot(pts.real, pts.imag, ".", ms=1.6, color="#1f77b4", label="Lugar das raízes")
        ol_p = np.roots(denL) if len(denL) > 1 else []
        ol_z = np.roots(ctl._limpa(numL)) if len(ctl._limpa(numL)) > 1 else []
        ax.plot(np.real(ol_p), np.imag(ol_p), "x", color="k", ms=8, mew=2, label="Polos MA")
        if len(ol_z):
            ax.plot(np.real(ol_z), np.imag(ol_z), "o", mfc="none", color="k", ms=7, mew=1.5, label="Zeros MA")
        zd = R["zd"]
        ax.plot([zd.real, zd.real], [zd.imag, -zd.imag], "*", color="red", ms=13, label="Polo desejado zd")
        if R["metodo"] in ("SINT", "OTIM"):
            ax.set_title("LR variando o ganho do Gc", fontsize=10)
        pmf = R["polos_mf"]
        ax.plot(pmf.real, pmf.imag, "s", mfc="none", color="#2ca02c", ms=8, mew=1.5, label="Polos MF")
        ax.axhline(0, color="#999", lw=0.5)
        ax.axvline(0, color="#999", lw=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal", adjustable="box")
        if R["metodo"] == "LR":
            ax.set_title("Lugar das Raízes no plano z")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imaginário")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    def _plot_bode(self, axm, axf, R):
        w = np.logspace(-3, math.log10(math.pi / R["T"]) + 0.5, 2000)
        curvas = [(R["num_s"], R["den_s"], "Planta G(s)", "#999999", "--")]
        if "num_Ls" in R:
            curvas.append((R["num_Ls"], R["den_Ls"], "C(s)·G(s)", "#1f77b4", "-"))
        for num, den, rot, cor, ls in curvas:
            H = ctl.avalia(num, den, 1j * w)
            axm.semilogx(w, 20 * np.log10(np.abs(H)), ls, color=cor, label=rot)
            axf.semilogx(w, np.degrees(ctl.fase_continua(num, den, w, R["Td"])), ls, color=cor)
        if "num_Ls" in R:
            axf.semilogx(w, np.degrees(ctl.fase_continua(R["num_Ls"], R["den_Ls"], w, R["T"] / 2 + R["Td"])), ":",
                         color="#d62728", label="C·G com atraso do ZOH")
        axm.axhline(0, color="k", lw=0.7)
        axf.axhline(-180, color="k", lw=0.7)
        wc = R.get("wc_alvo")
        if wc:
            for a in (axm, axf):
                a.axvline(wc, color="#2ca02c", lw=0.8, ls="--")
            axf.axhline(-180 + R["MF_alvo"], color="#2ca02c", lw=0.8, ls="--")
        axm.axvline(math.pi / R["T"], color="#d62728", lw=0.8, ls=":")
        axm.set_title("Diagrama de Bode (verde: ωc e MF alvo; vermelho: Nyquist π/T)", fontsize=9)
        axm.set_ylabel("Módulo (dB)")
        axf.set_ylabel("Fase (°)")
        axf.set_xlabel("ω (rad/s)")
        axm.grid(True, which="both", alpha=0.3)
        axf.grid(True, which="both", alpha=0.3)
        axm.legend(fontsize=7)
        axf.legend(fontsize=7)

    def _plot_degrau(self, ax, R):
        A = R["amp"]
        t, y = R["t"], R["y"] * A
        if "t_cont" in R:
            ax.plot(R["t_cont"], R["y_cont"] * A, color="#1f77b4", lw=1.2, label="Contínuo projetado (s)")
        if "y_des" in R:
            ax.step(t, R["y_des"] * A, where="post", color="#2ca02c", lw=1.2, ls="--", label="Desejada yd = T(z)")
        ax.step(t, y, where="post", color="#d62728", lw=1.4, label="Digital Gc(z)·G(z)")
        dc = R["dc"] * A if np.isfinite(R["dc"]) else np.nan
        if R["estavel"] and np.isfinite(dc):
            ax.axhline(dc, color="k", lw=0.7)
            ax.axhspan(dc * 0.98, dc * 1.02, color="#2ca02c", alpha=0.12, label="Faixa ±2%")
            ri = R["req_in"]
            if ri["PO"] is not None:
                ax.axhline(dc * (1 + ri["PO"] / 100), color="#ff7f0e", lw=0.8, ls="--", label="Overshoot máx.")
            if ri["ts"] is not None:
                ax.axvline(ri["ts"], color="#2ca02c", lw=0.8, ls="--", label="ts desejado")
            if ri["tp"] is not None:
                ax.axvline(ri["tp"], color="#9467bd", lw=0.8, ls="--", label="tp desejado")
            i = R["info"]
            if np.isfinite(i["PeakTime"]) and i["Overshoot"] > 0.5:
                ax.plot(i["PeakTime"], i["Peak"] * A, "o", color="#d62728", ms=5)
            tmax = max([v for v in (i["SettlingTime"], ri["ts"], ri["tp"], R["req"]["ts"])
                        if v is not None and np.isfinite(v)] + [5 * R["T"]]) * 1.6
            ax.set_xlim(0, min(tmax, t[-1]))
        ax.set_title(f"Degrau de {A:g} UE (malha fechada)", fontsize=9)
        ax.set_ylabel("Saída (UE)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")


if __name__ == "__main__":
    App().mainloop()
