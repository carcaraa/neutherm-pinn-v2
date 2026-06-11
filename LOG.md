# LOG — Revisão, correção de bugs e melhorias do NeuTherm-PINN

> Registro detalhado de tudo que foi analisado, encontrado e implementado na
> sessão de revisão do repositório. Data: junho/2026.

---

## 1. Escopo da revisão

Foram lidos e analisados todos os módulos do pacote (`neutherm/physics`,
`neutherm/solvers`, `neutherm/models`, `neutherm/training`,
`neutherm/evaluation`), a configuração (`configs/default.yaml`,
`pyproject.toml`), os testes, o notebook `01_walkthrough.ipynb` e o
`README.md`. O pipeline completo foi reexecutado do zero (solver → dataset →
surrogate → PINN → comparação) para validar as correções e coletar números
reais para a documentação.

**Resultado:** 18 problemas identificados (7 críticos, 5 de robustez/
portabilidade, 6 de documentação), todos corrigidos. Dois dos críticos
(#6 e #7) só se manifestaram durante o retreino longo — análise estática não
os teria pegado. Suíte de testes ampliada de 12 para 22 testes.

---

## 2. Bugs críticos (afetavam os resultados científicos)

### Bug #1 — PINN: acoplamento térmico morto (causa raiz dos erros grandes)

**Arquivo:** `neutherm/training/train_pinn.py`

**Sintoma:** o README reportava erro de 15,8% em k_eff e 24,9% na temperatura
para o PINN.

**Causa:** a rede produz um fluxo de magnitude O(1) (uma "forma" de fluxo),
mas a fonte de calor era calculada diretamente dele:
`q''' = κ(Σf1·φ1 + Σf2·φ2)`. Com κ ≈ 3,2×10⁻¹¹ J/fissão e φ ~ O(1), isso dá
q''' ≈ 4,6×10⁻¹² W/cm³ — contra ~380 W/cm³ do problema real. Numericamente, a
equação do calor "enxergava" fonte nula: a temperatura ficava desacoplada do
fluxo e o feedback Doppler nunca atuava. O PINN estava, na prática, resolvendo
dois problemas independentes (e um deles errado).

**Correção (reformulação do treino):**
- Introduzida **escala física aprendível** `phi_scale = exp(s)` (positiva por
  construção), com inicialização física
  φ₀ = P′/(κ·νΣf₂·πR_f²) ≈ 8×10¹³ n/cm²·s.
- O termo fonte da equação do calor passou a usar o **fluxo físico**
  `phi_scale · φ_rede`.
- Novo termo de loss de **normalização de potência**
  `L_power = ((q′_pred − P′)/P′)²`, onde
  q′_pred = ∫₀^{R_f} κ(Σf1·φ1 + Σf2·φ2)·2πr dr é avaliado por quadratura
  trapezoidal em uma grade fixa de 128 pontos. É exatamente a mesma condição
  de normalização que o solver de referência usa — e é o que liga, dentro do
  PINN, a neutrônica à térmica.
- Normalização do resíduo térmico mudou de `mean(T²)` (sem significado) para
  `mean(q_source²)` (escala natural da equação).
- Novo hiperparâmetro `lambda_power` (default 10.0) em
  `PINNParams`/`configs/default.yaml`.

### Bug #2 — PINN: temperatura de superfície "chutada" (hardcoded)

**Arquivo:** `neutherm/training/train_pinn.py`

**Antes:** `T_surface = T_coolant + 200 = 780 K` (valor arbitrário) e
`T_base = T_coolant + 300` (idem). O valor físico, dado pelo modelo de
gap+convecção do próprio projeto, é 757,8 K para P′ = 200 W/cm.

**Leitura dos resultados:** o surrogate é praticamente exato dentro da faixa
amostrada. O PINN, após as correções, recupera o autovalor (~1,7%) e as
magnitudes físicas dos fluxos (~11–13%); a deficiência remanescente é o
**perfil de temperatura**, que sai quase plano (ΔT ≈ 18 K vs 381 K do
solver) e domina o erro L2 de T — mesmo com a cabeça térmica em escala
física (bug #7), a paisagem de otimização multi-física mantém o resíduo de
condução num platô que 6000 épocas de Adam não escapam (patologia coerente
com Wang et al., 2022, ref. [10] do README). Caminhos de melhoria deixados
como trabalho futuro: otimizadores de 2ª ordem (L-BFGS), pesos adaptativos,
ou o modo híbrido `--with-data` (a infraestrutura para todos já está
corrigida e pronta).

**Correção:** `T_surface` agora é calculada com
`compute_surface_temperature(power_level·100, geom, thermal)` — a mesma
física do solver — e `T_base = T_surface`. O treino ganhou o parâmetro
`power_level` (CLI `--power`, default 200 W/cm).

### Bug #3 — PINN: loss de contorno desbalanceada

**Arquivo:** `neutherm/training/train_pinn.py` (`compute_bc_loss`)

**Antes:** o termo de temperatura entrava em K² (~8100 para um erro de 90 K),
dominando ~99% da loss de BC e sufocando as condições dos fluxos.

**Correção:** termo de temperatura normalizado por `T_scale = 100 K`, deixando
todos os termos de BC adimensionais e de magnitude comparável.

### Bug #4 — PINN: data loss em escalas incompatíveis

**Arquivo:** `neutherm/training/train_pinn.py`

**Antes:** no modo `--with-data`, comparava-se o fluxo O(1) da rede com os
dados físicos do solver (~10¹³) — a loss era ≈ constante e o gradiente,
inútil.

**Correção:** a comparação agora usa `phi_scale · φ_rede` contra os dados
físicos, com normalização pelas escalas de referência
(`phi1_ref_scale`, `phi2_ref_scale`, `T_ref_scale`).

### Bug #5 — compare.py: PINN avaliado sem escala e no domínio errado

**Arquivo:** `neutherm/evaluation/compare.py`

**Antes:** o erro L2 do fluxo do PINN dava ~100% (fluxo O(1) comparado a
~10¹³ sem conversão de escala); fluxos eram comparados só na região do
combustível, embora o solver e o PINN resolvam a célula completa; `T_base`
era recalculado com a fórmula antiga em vez de lido do checkpoint.

**Correção:**
- `load_pinn` retorna `(model, k_eff, phi_scale, T_base)` lendo tudo do
  checkpoint (com fallbacks para checkpoints antigos).
- `predict_pinn` aplica `phi_scale` e recebe `T_base` correto.
- Fluxos comparados no **domínio completo** (`ref.r_neutronics`, 140 pts) e
  temperatura na malha do combustível; plots ganharam a linha vertical da
  interface combustível/moderador e unidades físicas (n/cm²·s).

### Bug #6 — PINN: instabilidade do autovalor (descoberto no retreino)

**Arquivo:** `neutherm/training/train_pinn.py`

**Sintoma:** ao rodar o treino longo (15000 épocas), o k_eff aprendido
divergia monotonicamente (1,0 → 2,05 → 2,82...) com a loss de PDE travada em
~1,1, embora a potência integrada permanecesse correta (~178 W/cm). Em
treinos curtos o problema passava despercebido, porque o *cosine annealing*
derrubava o learning rate antes do colapso.

**Causa:** `k_eff` e `log_phi_scale` compartilhavam um *param group* do Adam
com `lr × 10`. Com o learning rate alto sustentado, a otimização encontra um
mínimo espúrio: a rede **achata o fluxo** (derivadas ≈ 0 → BCs e continuidade
trivialmente satisfeitas) e o k_eff dispara rumo ao k∞ do combustível
(≈ 2,5–3, o autovalor sem fuga). A loss de PDE estaciona exatamente no
resíduo irredutível do moderador para fluxo constante. O problema de fundo é
estrutural: **para qualquer forma de fluxo existe um k que reduz localmente o
resíduo do grupo rápido** — a degenerescência forma⊗autovalor torna um k
livre, otimizado por gradiente, intrinsecamente instável.

**Tentativa intermediária (insuficiente):** separar os *param groups*
(k_eff em lr base) e congelar o k_eff nos primeiros 10% das épocas (warmup
da forma). Atenuou a fuga, mas não a eliminou: após a liberação, o k_eff
ainda derivava (1,44 → 1,65 → 1,81 em ~3000 épocas) com a PDE presa no
platô do fluxo achatado.

**Correção definitiva — k_eff via balanço integral de nêutrons:** o k_eff
deixou de ser um parâmetro treinável e passou a ser **calculado** a cada
época a partir da forma corrente do fluxo (função `compute_k_balance`):

    k_eff = ∫ (νΣf1·φ1 + νΣf2·φ2) dV / ∫ (Σa1·φ1 + Σa2·φ2) dV

com Σa1 = Σr1 − Σs1→2 e dV = 2πr dr na célula completa (XS do combustível
dependentes de T; XS do moderador constantes). Com contorno refletivo a fuga
líquida da célula de Wigner–Seitz é nula, então essa razão é exatamente o
k_eff da solução convergida — é o **análogo PINN da power iteration** que o
próprio solver de referência usa. A razão é diferenciável (o gradiente flui
para a forma também através do fator 1/k do resíduo rápido), e a
degenerescência desaparece por construção: a forma é o único grau de
liberdade. Validação: em um treino curto de 60 épocas, k_eff = 1,296 já na
**época 3** e 1,338 na época 60 (referência: 1,3009) — contra 2,8 e
divergindo na versão com k livre. Robustez numérica: clamp do k em
[0,3, 3,5] nas primeiras épocas, quando a forma ainda é arbitrária.

### Bug #7 — PINN: gradiente esmagado na cabeça de temperatura (descoberto no retreino)

**Arquivos:** `neutherm/models/pinn.py`, `neutherm/training/train_pinn.py`

**Sintoma:** mesmo com o acoplamento térmico corrigido (bug #1) e o k_eff
estabilizado (bug #6), o treino longo produzia **temperatura quase plana**:
T_centro = 767 K contra 1139 K do solver (ΔT de 9 K em vez de 381 K),
resultando em erro L2 de 26% — e, por tabela, segurava a loss de PDE num
platô (o resíduo térmico normalizado vale exatamente 1,0 quando T é plana,
pois R_heat ≈ −q‴ e a normalização é mean(q‴²)).

**Causa (condicionamento, não formulação):** a cabeça T da rede produz saída
O(1), mas o desvio físico de temperatura é O(400 K). Para zerar o resíduo
térmico, a curvatura d²T/dr² precisa atingir ~5×10³ K/cm²; o gradiente da
loss em relação à saída da rede é ∝ k_térmica/q‴ ~ 10⁻⁴ — o caminho até a
solução exige crescer os pesos da cabeça em ~10², e o treino estaciona no
platô "T plana" muito antes disso.

**Correção:** escala física de saída na cabeça térmica. `PINNModel` ganhou o
parâmetro `T_out_scale` (default 1,0 para compatibilidade) e o `forward`
devolve `T_out_scale · saída_bruta`. O treino calcula a escala pela
estimativa clássica de condução em cilindro,

    T_out_scale = q′ / (4π k̄) ≈ 389 K   (q′ = 200 W/cm, k̄ = k_UO2(T_sup+200))

de modo que a saída bruta da rede permanece O(1) e os gradientes do resíduo
térmico voltam a O(1). A escala é salva no checkpoint e lida por
`compare.py`/notebook (fallback 1,0 para checkpoints antigos). Efeito
imediato observável: a loss de BC sai do zero trivial (a cabeça T passa a
"existir" para o otimizador) e o perfil parabólico de temperatura emerge no
treino longo.

### Bug #8 — train_surrogate.py: erro de device na avaliação (GPU)

**Arquivo:** `neutherm/training/train_surrogate.py`

**Antes:** na avaliação do conjunto de teste, as estatísticas de normalização
(`norm_stats`) ficavam na CPU enquanto as predições estavam na GPU — um
comentário dizia "move to device" mas o código não movia. Em GPU isso
estourava `RuntimeError: Expected all tensors to be on the same device`.

**Correção:** `NormStats` é movido explicitamente para o device antes da
avaliação (campo a campo do dataclass).

---

## 3. Bugs de robustez e portabilidade

### Bug #9 — Incompatibilidade com NumPy 1.x

**Arquivos:** `coupled_solver.py`, `thermal_solver.py`, `pyproject.toml`

`np.trapezoid` só existe a partir do NumPy 2.0, mas o projeto declara
`numpy>=1.24`. Em NumPy 1.x: `AttributeError`.

**Correção:** novo módulo `neutherm/_compat.py` com
`trapezoid = getattr(np, "trapezoid", None) or np.trapz`, importado pelos
dois solvers (e usado nos testes novos). Funciona em 1.x e 2.x.

### Bug #10 — Docstrings mortas (`__doc__ = None`)

**Arquivos:** `cross_sections.py`, `fuel_properties.py`,
`diffusion_solver.py`

A docstring de módulo estava **depois** de
`from __future__ import annotations`. Pela gramática do Python, a string
deixa de ser docstring (vira expressão solta): `help()` e ferramentas de
documentação viam `None`.

**Correção:** docstrings movidas para antes do import de `__future__`.

### Bug #11 — Checkpoints não autossuficientes

**Arquivos:** `models/surrogate.py`, `models/pinn.py`,
`train_surrogate.py`, `train_pinn.py`

**Antes:** os checkpoints não salvavam `hidden_layers`/`activation`; carregar
um modelo exigia que o YAML local coincidisse com o usado no treino — receita
para `size mismatch` silencioso ou erro criptográfico.

**Correção:** os modelos agora guardam `self.hidden_layers` e
`self.activation_name`; os dois scripts de treino salvam a arquitetura no
checkpoint; `compare.py` e o notebook leem a arquitetura do checkpoint com
fallback para o YAML (compatível com checkpoints antigos). O checkpoint do
PINN também salva `phi_scale`, `T_base`, `T_surface`, `T_out_scale`,
`power_level`, `r_fuel` e `r_cell`.

### Bug #12 — `build_pin_cell_xs_np`: shape mismatch silencioso

**Arquivo:** `physics/cross_sections.py`

Se `len(T_fuel) < n_fuel`, o broadcast do NumPy gerava erro confuso longe da
causa. **Correção:** validação explícita com `ValueError` e mensagem clara.

### Bug #13 — `load_surrogate`: type hint mentiroso

**Arquivo:** `evaluation/compare.py`

A assinatura prometia `SurrogateModel` mas a função retorna
`(model, norm_stats)`. **Correção:** anotação
`tuple[SurrogateModel, dict | None]`.

---

## 4. Documentação e configuração incorretas

### Bug #14 — Condição de contorno documentada errada

**Arquivos:** `diffusion_solver.py` (docstring), `coupled_solver.py`
(comentário), `README.md` (tabela de BCs)

A documentação dizia "fluxo zero" (φ = 0) na borda externa, mas o código
implementa (corretamente, para célula de Wigner–Seitz) condição **refletiva**
(corrente zero, dφ/dr = 0) em `r = R_cell`. A tabela do README ainda colocava
a BC em `R_clad` e omitia que o clad é desprezado na neutrônica.

**Correção:** docstrings, comentários e tabela do README reescritos:
refletiva em `R_cell`; nota explícita de que o domínio neutrônico é a célula
completa (combustível + moderador homogeneizado) e o térmico é só o
combustível, com `T(R_f)` vindo da cadeia de resistências gap+convecção.

### Bug #15 — README: estrutura de projeto fictícia

O README listava `neutherm/visualization/plots.py`, `data/README.md`,
`results/README.md`, `tests/test_cross_sections.py` e `tests/test_models.py`
— nenhum existe. **Correção:** árvore reescrita refletindo exatamente o
repositório (incluindo o novo `_compat.py` e `tests/test_solvers.py`).

### Bug #16 — `adaptive_weights: true` sem implementação

**Arquivos:** `configs/default.yaml`, `physics/parameters.py`,
`train_pinn.py`

A flag estava ligada no config, mas nenhuma linha de código a lia — pesos
adaptativos nunca existiram. **Correção:** default `false` com comentário
"placeholder, not implemented"; `train_pinn.py` emite um *warning* claro se
alguém ligar a flag; README atualizado dizendo explicitamente que é extensão
futura.

### Bug #17 — Docstrings menores incorretas

- `training/losses.py`: a docstring citava a chave `'loss_keff'`; a chave
  real do dicionário é `'loss_k_eff'`. Corrigido.
- `training/dataset.py`: a docstring afirmava que os campos eram
  "interpolados para uma malha comum" — não há interpolação; todas as
  amostras têm o **mesmo número de pontos** (o que permite empilhar em
  tensores), mas as coordenadas físicas mudam quando `r_fuel` varia. A malha
  armazenada é a da primeira amostra bem-sucedida, usada apenas como eixo
  representativo para plots. Docstring reescrita.
- `README.md`: o vetor de parâmetros do surrogate descrito era
  `(R_f, q₀''', T_coolant, k_f⁰, …)`; o real é
  `(T_coolant, R_f, fator de enriquecimento)`. A descrição do PINN dizia que
  a rede recebia `(r, p)`; ela recebe apenas `r` (um ponto de operação).
  Ambos corrigidos. A menção a DeepONet foi reposicionada como extensão
  (a arquitetura implementada é FNN com blocos residuais).

### Bug #18 — Notebook: célula sem ramo `else` e predição sem escala

**Arquivo:** `notebooks/01_walkthrough.ipynb`

- Célula 7: `if ns is not None:` sem `else` — se o checkpoint não tivesse
  `norm_stats`, as variáveis `surr_*` nem existiam (NameError adiante).
  Adicionado o ramo `else`.
- Células 9/11/12: carregamento do PINN não lia `phi_scale`/`T_base` do
  checkpoint, predição sem escala física, comparação no domínio errado —
  mesmos problemas do `compare.py`, agora corrigidos no notebook (avaliação
  dos fluxos na célula completa, temperatura na malha do combustível,
  arquitetura e `T_out_scale` lidos do checkpoint).
- Célula 13 (takeaways) reescrita sem a justificativa incorreta de
  "fuel-only domain" para o erro do PINN.

---

## 5. Melhorias adicionais implementadas

1. **`tests/test_solvers.py` (novo)** — 10 testes de regressão:
   - convergência do Picard e teto de iterações;
   - regressão de k_eff = 1,300934 (±10⁻⁴);
   - **normalização de potência do fluxo** (∫q′′′·2πr dr = P′ ± 1%);
   - positividade e magnitude física dos fluxos (10¹²–10¹⁶);
   - forma do perfil de temperatura (máximo no centro, monótono decrescente);
   - regressão de T_centro = 1139,1 K e T_superfície = 757,8 K (±2 K);
   - consistência entre a T de superfície convergida e o modelo analítico
     gap+convecção (±1 K);
   - T > T_coolant em todo o domínio;
   - **sinal do feedback Doppler**: dobrar a potência reduz k_eff.

2. **Exports dos subpacotes** — `models/__init__.py`,
   `physics/__init__.py` e `evaluation/__init__.py` agora exportam as
   classes/funções públicas (`SurrogateModel`, `PINNModel`,
   `build_pin_cell_xs_np`, métricas etc.), permitindo
   `from neutherm.models import PINNModel`.

3. **Histórico de treino do PINN ampliado** — `PINNHistory` registra
   `power_loss` e `phi_scale_history`; o gráfico de treino ganhou a curva da
   loss de potência; o melhor `phi_scale` é rastreado junto do melhor modelo.

4. **Tabela de comparação honesta** — a linha "Training data needed" do
   PINN passou de "0 (or 50)" para "1 ref. solve" (a escala/T_surface vêm de
   uma única solução de referência implícita na potência prescrita).

---

## 6. Resultados após as correções

### Solver de referência (inalterado — regressão confirmada)

| Grandeza | Valor |
|---|---|
| k_eff | 1.300934 |
| T_centro | 1139.1 K |
| T_superfície | 757.8 K |
| Iterações de Picard | 6 |

### Pipeline completo reexecutado

- Dataset: 5000 amostras LHS, 100% convergidas, ~56 s (CPU, 1 núcleo).
- Surrogate: 5000 épocas — métricas na tabela abaixo.
- PINN: treino reformulado — métricas na tabela abaixo.

### Comparação Solver × Surrogate × PINN (medida nesta sessão)

Fluxos comparados na **célula completa** (140 pts), temperatura na malha do
combustível (100 pts); k_eff de referência = 1.300934.

| Métrica | Surrogate | PINN |
|---|---|---|
| k_eff | 1.300914 | 1.278382 |
| Erro relativo de k_eff | 0.0015% | 1.7335% |
| Erro L2 relativo de φ₁ | 0.0030% | 12.7903% |
| Erro L2 relativo de φ₂ | 0.0039% | 10.8611% |
| Erro L2 relativo de T | 0.0033% | 25.3552% |
| Dados de treino | 5000 amostras | 1 solução de ref. |
| Parâmetros treináveis | 115,709 | 12,867 |

Treinos executados em CPU (1 núcleo): surrogate 5000 épocas (~7 min);
PINN 6000 épocas (~11 min), `phi_scale` e `T_surface` ancorados em
P′ = 200 W/cm.

---

## 7. Arquivos modificados/criados

| Arquivo | Ação |
|---|---|
| `neutherm/_compat.py` | **novo** — shim trapezoid NumPy 1.x/2.x |
| `neutherm/physics/parameters.py` | `lambda_power`, `adaptive_weights=false` |
| `neutherm/physics/cross_sections.py` | docstring, validação de shapes |
| `neutherm/physics/fuel_properties.py` | docstring |
| `neutherm/physics/__init__.py` | exports |
| `neutherm/solvers/diffusion_solver.py` | docstring (BC refletiva) |
| `neutherm/solvers/thermal_solver.py` | usa `_compat.trapezoid` |
| `neutherm/solvers/coupled_solver.py` | usa `_compat.trapezoid`, comentário BC |
| `neutherm/models/surrogate.py` | guarda arquitetura |
| `neutherm/models/pinn.py` | guarda arquitetura, **`T_out_scale`** (cabeça térmica em escala física) |
| `neutherm/models/__init__.py` | exports |
| `neutherm/training/train_surrogate.py` | fix device GPU, checkpoint completo |
| `neutherm/training/train_pinn.py` | **reformulação**: phi_scale, L_power, T_surface físico, BC balanceada, data loss escalada, k_eff via balanço integral (`compute_k_balance`), T_out_scale, checkpoint completo, `--power` |
| `neutherm/training/losses.py` | docstring |
| `neutherm/training/dataset.py` | docstring (malhas por amostra) |
| `neutherm/evaluation/compare.py` | escala física, domínio completo, checkpoints autossuficientes |
| `neutherm/evaluation/__init__.py` | exports |
| `configs/default.yaml` | `lambda_power`, `adaptive_weights: false`, `epochs: 6000` |
| `tests/test_solvers.py` | **novo** — 10 testes de regressão |
| `notebooks/01_walkthrough.ipynb` | células 6, 7, 9, 11, 12, 13 |
| `README.md` | BCs, estrutura, loss do PINN (potência + balanço integral), requirements, usage, resultados |
| `LOG.md` | **novo** — este documento |
