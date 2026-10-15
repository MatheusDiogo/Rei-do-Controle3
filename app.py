"""
Rei do Controle - Projeto de Controlador Digital Gc(z)

Interface gráfica (tkinter + matplotlib) sobre o núcleo de cálculo em controle.py.
"""
import math
import tkinter as tk
from tkinter import ttk, messagebox

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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(TITULO)
        self.geometry("1360x820")
        self.minsize(1100, 680)
        self.resultado = None
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

    def _monta_entrada(self):
        esq = ttk.Frame(self, padding=10)
        esq.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(esq, text="Rei do Controle", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(esq, text="Projeto de controlador digital Gc(z)", style="Dica.TLabel").pack(anchor="w", pady=(0, 8))

        # Planta
        fp = ttk.LabelFrame(esq, text=" Planta ", padding=8)
        fp.pack(fill=tk.X, pady=4)
        ttk.Label(fp, text="G(s) =").grid(row=0, column=0, sticky="w")
        self.var_G = tk.StringVar(value=EXEMPLO_G)
        eg = ttk.Entry(fp, textvariable=self.var_G, width=42, font=FONTE_MONO)
        eg.grid(row=1, column=0, columnspan=2, sticky="we", pady=2)
        ttk.Label(fp, text="Ex: 10*(s+2)/((s+1)*(s+5))  ·  2*e^(-0.5*s)/(3*s+1)\n"
                           "Multiplicação somente com * ; potência com ^ ; '.' = decimal.",
                  style="Dica.TLabel").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(fp, text="Verificar G(s)", command=self.verifica_planta).grid(row=3, column=0, sticky="w", pady=(4, 0))
        self.lbl_planta = ttk.Label(fp, text="", font=("Consolas", 9), foreground="#1a5fb4", justify=tk.LEFT)
        self.lbl_planta.grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Label(fp, text="Período de amostragem T (ms):").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.var_T = tk.StringVar(value="100")
        ttk.Entry(fp, textvariable=self.var_T, width=10).grid(row=5, column=1, sticky="e", pady=(8, 0))
        fp.columnconfigure(0, weight=1)

        # Requisitos
        fr = ttk.LabelFrame(esq, text=" Requisitos (deixe em branco o que não for pedido) ", padding=8)
        fr.pack(fill=tk.X, pady=4)
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
                        variable=self.var_erro).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        fr.columnconfigure(0, weight=1)

        # Método
        fm = ttk.LabelFrame(esq, text=" Método de projeto ", padding=8)
        fm.pack(fill=tk.X, pady=4)
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
                                                                                    sticky="w", pady=(6, 0))
        self.lbl_folga = ttk.Label(fm, text="Folga de fase (°):")
        self.lbl_folga.grid(row=5, column=0, sticky="w", padx=(22, 0))
        self.var_folga = tk.StringVar(value="5")
        self.ent_folga = ttk.Entry(fm, textvariable=self.var_folga, width=8)
        self.ent_folga.grid(row=5, column=1, sticky="e")
        self.var_tia = tk.BooleanVar(value=True)
        ttk.Checkbutton(fm, text="Compatível com blocos GZ1/GZ2 do TIA (matched)",
                        variable=self.var_tia).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        fm.columnconfigure(0, weight=1)

        # Botões
        fb = ttk.Frame(esq)
        fb.pack(fill=tk.X, pady=(10, 4))
        ttk.Button(fb, text="CALCULAR", style="Calc.TButton", command=self.calcula).pack(fill=tk.X)
        fb2 = ttk.Frame(esq)
        fb2.pack(fill=tk.X)
        ttk.Button(fb2, text="Copiar Gc(z)", command=self.copia_gc).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(fb2, text="Copiar relatório", command=self.copia_relatorio).pack(side=tk.LEFT, expand=True,
                                                                                    fill=tk.X, padx=4)
        ttk.Button(fb2, text="Limpar", command=self.limpa).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.lbl_status = ttk.Label(esq, text="Preencha os dados e clique em CALCULAR.", wraplength=330,
                                    style="Dica.TLabel")
        self.lbl_status.pack(anchor="w", pady=6)
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
        self.ent_folga.configure(state=tk.DISABLED if lr else tk.NORMAL)
        self.lbl_folga.configure(state=tk.DISABLED if lr else tk.NORMAL)

    @staticmethod
    def _set_text(w, txt):
        w.configure(state=tk.NORMAL)
        w.delete("1.0", tk.END)
        w.insert("1.0", txt)
        w.configure(state=tk.DISABLED)

    # ------------------------------------------------------------ ações
    def verifica_planta(self):
        try:
            num, den, Td = ctl.parse_planta(self.var_G.get())
            polos = np.roots(den) if len(den) > 1 else []
            txt = planta_str(num, den, Td)
            txt += "\nPolos: " + ", ".join(ctl._c(p) if abs(p.imag) > 1e-12 else f"{p.real:.5g}" for p in polos)
            self.lbl_planta.configure(text=txt, foreground="#1a5fb4")
        except ctl.ErroProjeto as e:
            self.lbl_planta.configure(text=str(e), foreground="#c01c28")

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
            self.config(cursor="watch")
            self.update_idletasks()
            R = ctl.projeta(self.var_G.get(), T, PO, tp, ts, self.var_erro.get(),
                            self.var_met.get(), self.var_sub.get(), folga, tia=self.var_tia.get())
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
        linhas.append(("Esforço de controle máx |u| (V)", "—", _fmt(R["u_max"]), ("—", "")))
        linhas.append(("Esforço de controle em regime u(∞) (V)", "—", _fmt(R["u_regime"]), ("—", "")))
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
        L = []
        sep = "=" * 78
        L += [sep, "  REI DO CONTROLE - MEMÓRIA DE CÁLCULO", sep, ""]
        L += ["[ PLANTA ]", planta_str(R["num_s"], R["den_s"], R["Td"]), ""]
        if R["Td"] > 0:
            L += [f"Atraso de transporte: Td = {R['Td']:g} s = {R['d_atraso']}·T + {R['theta_atraso']:.6g} s", ""]
        L += [f"Discretização ZOH com T = {R['T'] * 1000:g} ms:", fracao("G(z)", R["num_z"], R["den_z"]), ""]
        pz = np.roots(R["den_z"]) if len(R["den_z"]) > 1 else []
        n0 = sum(1 for p in pz if abs(p) < 1e-9)
        lista = [f"{ctl._c(p)} (|p|={abs(p):.4f})" for p in pz if abs(p) >= 1e-9]
        if n0:
            lista.append(f"0 (x{n0}{', vindos do atraso' if R['Td'] > 0 else ''})")
        L += ["Polos de G(z): " + ", ".join(lista), ""]
        q = R["req"]
        L += ["[ REQUISITOS ]"]
        ri = R["req_in"]
        val = {k: ("—" if ri[k] is None else f"{ri[k]:g}") for k in ("PO", "tp", "ts")}
        L += [f"  Entrada: PO = {val['PO']} %, tp = {val['tp']} s, ts(2%) = {val['ts']} s, "
              f"erro nulo = {'sim' if ri['erro_nulo'] else 'não'}"]
        L += [f"  ζ = {q['zeta']:.4f}    ωn = {q['wn']:.4f} rad/s    ωd = {q['wd']:.4f} rad/s    σ = {q['sigma']:.4f}"]
        L += [f"  2ª ordem teórica: PO = {q['PO']:.2f} %, tp = {q['tp']:.4f} s, ts = {q['ts']:.4f} s"]
        for n in q["notas"]:
            L += ["  " + n]
        L += [""]
        met = ("Lugar das Raízes - " + NOMES_LR.get(R["sub_metodo"], R["sub_metodo"])
               if R["metodo"] == "LR" else "Resposta em Frequência - Avanço de fase + "
               + ("matched" if R["tia"] else "Tustin"))
        L += [f"[ PROJETO: {met} ]"]
        L += ["  " + p for p in R["passos"]]
        L += [""]
        if R["metodo"] == "RF" and "num_cs" in R:
            L += ["Controlador contínuo:", fracao("C(s)", R["num_cs"], R["den_cs"], "s"), ""]
        L += ["[ CONTROLADOR DIGITAL ]", texto_controlador(R), "", texto_tia(R), ""]
        L += ["[ MALHA FECHADA ]", fracao("T(z)", R["num_mf"], R["den_mf"]), ""]
        L += ["  Polos: " + ", ".join(f"{ctl._c(p)} (|p|={abs(p):.4f})" for p in R["polos_mf"])]
        i = R["info"]
        L += ["", "[ STEPINFO - RESULTADO EM MALHA FECHADA DIGITAL ]",
              f"  Sobressinal       : {_fmt(i['Overshoot'], 4)} %",
              f"  Tempo de pico     : {_fmt(i['PeakTime'])} s",
              f"  Tempo de subida   : {_fmt(i['RiseTime'])} s",
              f"  Tempo de acomod.  : {_fmt(i['SettlingTime'])} s",
              f"  Valor de regime   : {_fmt(R['dc'])}",
              f"  Erro estacionário : {_fmt(R['ess'], 6)}",
              f"  Esforço máx |u|   : {_fmt(R['u_max'])} V   (u em regime = {_fmt(R['u_regime'])} V)"]
        if R["metodo"] == "RF" and "margens_s" in R:
            mo, ms, mz = R["margens_orig"], R["margens_s"], R["margens_s_zoh"]
            L += ["", "[ MARGENS (BODE) ]",
                  f"  Planta original   : MF = {_fmt(mo['Pm'], 2)}° em ωc = {_fmt(mo['Wcp'])} rad/s, "
                  f"MG = {_fmt(mo['Gm_dB'], 2)} dB",
                  f"  Compensado (cont.): MF = {_fmt(ms['Pm'], 2)}° em ωc = {_fmt(ms['Wcp'])} rad/s, "
                  f"MG = {_fmt(ms['Gm_dB'], 2)} dB",
                  f"  Compensado + ZOH  : MF = {_fmt(mz['Pm'], 2)}°, MG = {_fmt(mz['Gm_dB'], 2)} dB"]
            if "info_cont" in R:
                ic = R["info_cont"]
                L += [f"  Contínuo (teórico): PO = {_fmt(ic['Overshoot'], 2)} %, tp = {_fmt(ic['PeakTime'])} s, "
                      f"ts = {_fmt(ic['SettlingTime'])} s"]
        if len(R["candidatos"]) > 1:
            L += ["", "[ VARIANTES AVALIADAS ]  (a de menor custo é usada se a clássica falhar)"]
            for c in R["candidatos"]:
                if "falha" in c:
                    L += [f"  - {c['rotulo']}: inviável ({c['falha']})"]
                    continue
                ci = c["info"]
                marca = "  <== usada" if c is R["escolhido"] else ""
                if c["estavel"]:
                    L += [f"  - {c['rotulo']}: PO = {_fmt(ci['Overshoot'], 2)} %, tp = {_fmt(ci['PeakTime'], 3)} s, "
                          f"ts = {_fmt(ci['SettlingTime'], 3)} s, custo = {c['custo']:.3f}{marca}"]
                else:
                    L += [f"  - {c['rotulo']}: instável{marca}"]
        if R["avisos"]:
            L += ["", "[ AVISOS ]"] + ["  • " + a for a in R["avisos"]]
        self.txt_mem.delete("1.0", tk.END)
        self.txt_mem.insert("1.0", "\n".join(L))

    def _mostra_graficos(self, R):
        fig = self.fig
        fig.clear()
        gs = fig.add_gridspec(2, 2)
        if R["metodo"] == "LR":
            self._plot_lgr(fig.add_subplot(gs[:, 0]), R)
        else:
            axm = fig.add_subplot(gs[0, 0])
            self._plot_bode(axm, fig.add_subplot(gs[1, 0], sharex=axm), R)
        ax_y = fig.add_subplot(gs[0, 1])
        self._plot_degrau(ax_y, R)
        self._plot_esforco(fig.add_subplot(gs[1, 1], sharex=ax_y), R)
        fig.tight_layout()
        self.canvas.draw()

    def _plot_esforco(self, ax, R):
        t, u = R["t"], R["u"]
        ax.step(t, u, where="post", color="#9467bd", lw=1.4, label="u = T(z)/G(z) · degrau")
        if np.all(np.isfinite(u)) and len(u):
            i = int(np.argmax(np.abs(u)))
            ax.plot(t[i], u[i], "o", color="#d62728", ms=5)
            ax.annotate(f"máx |u| = {abs(u[i]):.4g} V  (t = {t[i]:.3g} s)", (t[i], u[i]),
                        textcoords="offset points", xytext=(8, -12 if u[i] > 0 else 8), fontsize=8,
                        color="#d62728")
            if np.isfinite(R["u_regime"]):
                ax.axhline(R["u_regime"], color="k", lw=0.7, ls="--",
                           label=f"u em regime = {R['u_regime']:.4g} V")
        ax.set_title("Esforço de controle (degrau unitário na referência)", fontsize=10)
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
        pmf = R["polos_mf"]
        ax.plot(pmf.real, pmf.imag, "s", mfc="none", color="#2ca02c", ms=8, mew=1.5, label="Polos MF")
        ax.axhline(0, color="#999", lw=0.5)
        ax.axvline(0, color="#999", lw=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal", adjustable="box")
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
        t, y = R["t"], R["y"]
        if "t_cont" in R:
            ax.plot(R["t_cont"], R["y_cont"], color="#1f77b4", lw=1.2, label="Contínuo projetado (s)")
        ax.step(t, y, where="post", color="#d62728", lw=1.4, label="Digital Gc(z)·G(z)")
        dc = R["dc"]
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
                ax.plot(i["PeakTime"], i["Peak"], "o", color="#d62728", ms=5)
            # janela de tempo: até ~1,5x o maior entre ts obtido e desejado
            tmax = max([v for v in (i["SettlingTime"], ri["ts"], ri["tp"], R["req"]["ts"])
                        if v is not None and np.isfinite(v)] + [5 * R["T"]]) * 1.6
            ax.set_xlim(0, min(tmax, t[-1]))
        ax.set_title("Resposta ao degrau unitário em malha fechada", fontsize=10)
        ax.set_ylabel("Saída")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")


if __name__ == "__main__":
    App().mainloop()
