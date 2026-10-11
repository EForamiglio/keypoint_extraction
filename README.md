# keypoint_extraction

Aplicação standalone para capturar keypoints de pose humana (YOLO pose) e
organizá-los em cenários de teste rotulados, para gerar dados de
validação/retreino do classificador GRU de detecção de quedas do PFC.

Só captura — não roda o classificador nem gera alertas. O rótulo do cenário é
escolhido **antes** de gravar, de um catálogo predefinido, então nunca precisa
ser reconstruído depois a partir de nomes de arquivo ou memória.

## Instalação

Requer Python 3.10+ e uma webcam.

```bash
git clone git@github.com:EForamiglio/keypoint_extraction.git
cd keypoint_extraction
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

O modelo `yolov8n-pose.pt` é baixado automaticamente pelo Ultralytics na
primeira execução (~6.5 MB) — é necessário ter internet na primeira vez que o
app rodar. Depois disso ele fica salvo localmente (arquivo `.pt` na raiz do
projeto, ignorado pelo Git).

## Como rodar

```bash
python main.py
```

Flags disponíveis (todas opcionais):

| Flag | Padrão | Uso |
|---|---|---|
| `--camera-index` | `0` | Índice da câmera. Use `1`, `2`... se tiver mais de uma webcam |
| `--model` | `yolov8n-pose.pt` | Outro modelo YOLO pose, se quiser comparar |
| `--conf` | `0.4` | Confiança mínima do YOLO para considerar uma detecção (era 0.25: deixava passar caixas duplicadas/espúrias) |
| `--iou` | `0.5` | Limiar de IoU do NMS. O padrão do YOLO (0.7) mantém caixas duplicadas da mesma pessoa com IoU ~0.66 |
| `--imgsz` | `640` | Resolução de entrada do YOLO |
| `--device` | `cpu` | `cpu` ou `cuda` (se tiver GPU NVIDIA configurada) |
| `--data-dir` | `data/` | Onde `captures/` e `manifest.csv` são gravados |
| `--scenarios` | `scenarios.yaml` | Caminho de um catálogo alternativo |
| `--conf-threshold` | `0.25` | Abaixo disso, um keypoint individual é zerado |
| `--min-frames-per-person` | `10` | Pessoas com menos frames válidos na tomada são descartadas |
| `--countdown` | `3` | Segundos de contagem regressiva antes de gravar |

Exemplos:

```bash
# webcam secundária
python main.py --camera-index 1

# gravar em outra pasta (ex: um pendrive, ou uma pasta por sessão de captura)
python main.py --data-dir ./data/sessao_2026_09_15
```

## Fluxo de uso

```
1. O app carrega o YOLO pose (só uma vez).
2. Aparece o menu de cenários, agrupado, com o próximo número de take de cada um.
3. Você digita o número do cenário e aperta ENTER. A câmera é aberta nesse momento
   (e fechada ao fim da tomada — segurá-la aberta parada no menu fazia o Windows
   invalidar o dispositivo depois de alguns minutos).
4. Uma contagem regressiva de --countdown segundos te dá tempo de se posicionar;
   durante ela o app já lê frames para estabilizar a exposição.
5. A janela de gravação abre com o esqueleto sobreposto (só pra você conferir
   qualidade da captura — nada é classificado).
     [SPACE] ou [s]  -> para a gravação e SALVA
     [ESC]            -> aborta e DESCARTA (nada é gravado em disco)
6. Volta automaticamente ao menu, pronto pra próxima tomada.
```

Exemplo de sessão de terminal:

```
Cenários disponíveis:
  [1] queda_lado_esquerdo         (FALL)   próximo take: 01
  [2] queda_lado_direito          (FALL)   próximo take: 01
  [3] queda_costas_esquerdo       (FALL)   próximo take: 01
  [4] queda_costas_direito        (FALL)   próximo take: 01
  [5] queda_de_frente             (FALL)   próximo take: 01
  [6] andando                     (ADL)    próximo take: 01
  [7] sentando                    (ADL)    próximo take: 01
  ...
  [q] sair
> 7
Abrindo câmera...
[janela de captura abre — pressione SPACE ao terminar a tomada]
Salvo: 1 clipe(s) -> ['sentando_take01_p0']  (19.6s a 18.1 fps)

Cenários disponíveis:
  [1] queda_lado_esquerdo         (FALL)   próximo take: 01
  ...
  [7] sentando                    (ADL)    próximo take: 02
  [r] repetir último cenário usado
  [q] sair
> r
Gravando 'sentando' em 1...
...
```

Repare que o "próximo take" de `sentando` já sobe pra `02` sozinho, e `[r]`
deixa repetir o mesmo cenário sem digitar o número de novo — pensado pra
gravar várias tomadas seguidas do mesmo cenário rapidamente.

## Como adicionar um cenário novo

Edite `scenarios.yaml` e acrescente uma entrada:

```yaml
  - id: subindo_escada
    label: ADL
    description: "Subindo um lance de escada"
```

- `id`: só minúsculas, números e `_` (vira parte do nome dos arquivos gravados
  — **nunca renomeie um `id` depois de já existirem dados gravados com ele**,
  isso quebra a rastreabilidade das gravações anteriores).
- `label`: `ADL` ou `FALL` — só isso, é o que decide o valor gravado em
  `_labels.npy` (0 para ADL, 2 para FALL — ver seção abaixo).
- `description`: opcional, só aparece no menu.

Nada em código muda — o app relê `scenarios.yaml` a cada execução.

## Onde os dados ficam e o que significam

```
data/
├── captures/
│   └── <scenario_id>/
│       ├── <scenario_id>_take01_p0.npy          # (T, 17, 3) x,y,conf em pixels, COCO-17
│       ├── <scenario_id>_take01_p0_labels.npy   # (T,) — 0=ADL ou 2=FALL, uniforme na tomada
│       └── <scenario_id>_take01_p0_t.npy        # (T,) segundos desde o 1º frame (float64)
└── manifest.csv
```

`manifest.csv` tem uma linha por clipe gravado:

| Coluna | O que é |
|---|---|
| `clip_id` | Identificador único do clipe (`<scenario_id>_take<NN>_p<i>`) |
| `source` | Constante `"keypoint_extraction"` — identifica a origem do dado |
| `is_fall` | `True`/`False`, derivado do `label` do cenário |
| `src_fps` | **FPS efetivo medido na tomada** (frames ÷ tempo decorrido), não o que o driver reporta — ver nota abaixo |
| `px_path` / `lbl_path` | Caminhos absolutos dos `.npy` correspondentes |
| `group_id` | `<scenario_id>_take<NN>` — mesma tomada = mesmo group_id, útil para não misturar pessoas do mesmo take entre splits de treino/validação no futuro |
| `scenario_id` / `scenario_label` | Igual ao `id`/`label` de `scenarios.yaml` |
| `take_number` | Número da tomada dentro do cenário |
| `person_idx` | Índice da pessoa dentro da tomada (`0`, `1`, ...) — ver seção seguinte |
| `n_frames` | Quantos frames a tomada teve |
| `session_id` | Identifica a execução do app em que o clipe foi gravado |
| `recorded_at` | Timestamp UTC |
| `operator_note` | Reservado para anotação manual futura (hoje sempre vazio) |

**Por que o fps é medido e não nominal.** O app grava um frame por inferência do
YOLO, então a taxa real fica abaixo do que a câmera reporta (30): medimos de 12 a
18 fps conforme a máquina e a iluminação (a câmera reduz a taxa em pouca luz). Por
isso cada clipe tem o `_t.npy` com o instante de cada frame. **Qualquer reamostragem
(ex.: para 10 fps) deve ser feita por tempo usando o `_t.npy`, nunca por índice de
frame assumindo 30 fps** — isso infla as features de velocidade.

`manifest.csv` é append-only: cada tomada salva acrescenta linha(s) novas,
nunca reescreve as anteriores. Interromper o app (Ctrl+C) entre uma tomada e
outra não perde o que já foi salvo.

## Cenários com mais de uma pessoa

Cenários como `duas_pessoas_andando` ou `abracando` não precisam de nenhuma
configuração especial: cada pessoa que o tracker do YOLO identificar durante a
tomada vira um clipe separado automaticamente (`_p0`, `_p1`, ...), com o mesmo
cenário, rótulo e número de take. Uma tomada com 2 pessoas em quadro rende 2
clipes de treino.

## Troubleshooting

- **"Não foi possível abrir a câmera de índice 0"** — outra aplicação pode
  estar usando a câmera (Zoom, Teams, o navegador), ou o índice está errado.
  Tente `--camera-index 1`.
- **"Erro de câmera ... dispositivo desconectado?"** com `MSMF ... -1072873822` no
  log — o Windows invalidou o dispositivo (USB suspenso, outro app pegou a câmera).
  O app volta ao menu sem gravar; escolha o cenário de novo (a câmera é reaberta).
  Se persistir, reconecte o cabo USB.
- **Apareceram clipes `_p1`, `_p2` num take com uma pessoa só** — são detecções
  espúrias de objetos da cena. O app já descarta tracks sem tronco (ombros e quadris)
  e usa `--conf 0.4`/`--iou 0.5`; se ainda ocorrer, tire objetos do quadro ou suba
  `--conf`. Para avaliar, use o track com mais frames de tronco válido.
- **Download do modelo falha** — precisa de internet na primeira execução.
  Depois disso, o `.pt` fica salvo localmente e não precisa mais baixar.
- **Uma pessoa não aparece no manifest depois de gravar** — provavelmente
  ficou com menos de `--min-frames-per-person` frames válidos na tomada
  (track curto/instável) e foi descartada de propósito, pra não poluir o
  dataset com tracks fantasma.
- **Quero confirmar que `data/` não vai pro Git** — rode `git status` depois
  de gravar uma tomada; `data/` não deve aparecer como arquivo não rastreado
  (está no `.gitignore`).

## Roteiro de smoke test

Depois de instalar, antes de uma campanha de gravação real:

1. `pytest tests/` — todos os testes devem passar, sem precisar de câmera.
2. `python main.py --help` — confirma que a CLI está funcionando.
3. `python main.py` — grave 1 take de um cenário `FALL` e 1 de um `ADL`.
4. Confira `data/captures/<scenario_id>/` — devem existir `<clip_id>.npy`
   (shape `(T,17,3)`) e `<clip_id>_labels.npy` (shape `(T,)`).
5. Confira `data/manifest.csv` — deve ter uma linha por clipe gravado, com as
   colunas descritas acima preenchidas.
6. Grave uma tomada e aperte `ESC` no meio — confirme que **nada** foi
   adicionado a `data/captures/` nem a `data/manifest.csv`.
7. (Opcional) grave um cenário com 2 pessoas em quadro (ex.
   `duas_pessoas_andando`) e confirme 2 clipes (`_p0`, `_p1`) no manifest.

> **Nota:** os passos 3-7 exigem webcam e interação manual (escolher opções,
> apertar teclas) — não são automatizáveis. Se algo neste roteiro não bater
> com o comportamento real do app, isso é bug — abra uma issue ou ajuste este
> README junto com a correção.

---

## Notas para continuidade (outro chat / assistente)

Esta seção existe para quem retomar o trabalho em outra máquina sem o histórico da
conversa original. O contexto abaixo vale em 2026-10-11.

### Contexto

PFC de detecção de quedas com visão computacional (YOLO pose → classificador). Quatro
repositórios irmãos, todos na mesma pasta pai:

| Pasta | Papel |
|---|---|
| `keypoint_extraction/` (este) | Captura keypoints rotulados por cenário. Só captura, não classifica |
| `fall-classifier-bench/` | Avaliação offline dos modelos sobre os clipes capturados (branch `avaliacao-cenarios-campo`) |
| `modelos/` | Treino e preparação de dados do GRU (branch `treinamento_maisDados_e_sinteticos`) |
| `vigia/` | Código de deploy do colega Pedro Reis. **Não alterar**; o modelo matemático dele está na branch `develop`, em `vigia-fall/capture/classifiers/math_classifier.py` |

Modelos em comparação: **GRU 34f** (`gru_2classes.onnx`, o do deploy), **GRU 37f**
(`gru_2classes_v2.onnx`, acrescenta 3 canais de velocidade) e o **modelo matemático**
do Pedro (Kalman + features geométricas + score ponderado + máquina de estados).

### Objetivo desta rodada

Regravar o conjunto de teste em ambiente controlado: 16 cenários do `scenarios.yaml`,
10 takes por cenário, para reavaliar os três modelos. A campanha anterior foi gravada
sem timestamps e com clipes curtos demais, o que limitou o que dava para concluir.

### Protocolo de gravação recomendado

Derivado de um teste com 1 queda e 1 sentando (não de uma amostra estatística):

- **Clipes de ~10 s**: ~3 s antes (em pé ou na atividade normal), ~2 s de evento, ≥5 s
  depois (deitado/parado). Medido: o take mais curto em que os três modelos ainda
  alertaram foi o de 8 s (queda começando em ~4,3 s). O modelo mais lento precisou de
  ~3 s após o início da queda.
- Hoje a gravação só para com **SPACE** — a duração é manual. Tente manter ~10 s.
- **Só quem faz o cenário no quadro**, exceto `abracando` e `duas_pessoas_andando`.
- **Iluminação fixa e forte, câmera fixa.** Em pouca luz a câmera reduz a taxa de frames.
- Confira o fps impresso ao salvar (`19.6s a 18.1 fps`). O esperado é algo entre 12 e 20.

### Armadilhas já conhecidas

- **fps varia por máquina e por luz** (medido: 12,1 / 18,0 / 22,4). Daí o `_t.npy` por
  clipe. Reamostre **por tempo**, nunca por índice assumindo 30 fps.
- `px_path` / `lbl_path` do `manifest.csv` são **caminhos absolutos da máquina que
  gravou** e não servem em outra. Localize os arquivos por convenção de pasta
  (`captures/<scenario_id>/<clip_id>.npy`), como faz `fall-classifier-bench/captures_dataset.py`.
- Em Windows, `cv2.VideoCapture` lista também a **OBS Virtual Camera** como um índice
  (mostra a tela de logo do OBS). Confira qual índice é a câmera real antes de gravar.
- No **macOS**, o terminal pode pedir permissão de câmera na primeira execução; o índice
  da câmera embutida costuma ser `0` e o de uma USB externa, `1`.
- O tracker pode perder o tronco por ~0,4 s no pico da queda (aparece como lacuna na
  curva do ângulo). Não impediu a detecção no teste, mas vale observar.
- `data/` está no `.gitignore`. **Os dados gravados não vão para o Git.**

### Como trazer os dados de volta

Zipe `data/captures/` e `data/manifest.csv` (não a `.venv`) e leve para a máquina onde
roda a análise. Para zipar só o código: `git archive --format=zip -o saida.zip HEAD`.

### Pendências e sugestões (nada disso está implementado)

1. **`--duration N`**: parada automática após N segundos, para padronizar o tamanho dos
   clipes e dispensar quem aperta SPACE. Mostrar o tempo restante no HUD.
2. **Marcar o início da queda.** Os rótulos são constantes por clipe, e num take de 19 s
   só ~5 s são a queda. Dá para derivar o início da curva do tronco (tronco > 60° por 2+
   frames; pré-queda = 1 s antes, a mesma regra do treino) e habilitar métricas por janela
   e tempo até o alerta.
3. **Adaptar a avaliação para reamostrar por tempo.** `fall-classifier-bench/captures_dataset.py`
   hoje faz `assert src_fps == 30` e reamostra por índice, então **recusa os dados novos**.
   `fall-classifier-bench/analyze_new_takes.py::resample_by_time` já faz do jeito certo.
4. **Refazer a comparação GRU 34f × 37f com reamostragem por tempo.** A conclusão
   anterior ("o 37f é significativamente pior", McNemar p = 0,009) foi obtida assumindo
   30 fps e está **provavelmente contaminada**: o erro de tempo infla as features de
   velocidade do 37f (1,15x a 1,36x medido nos takes de teste; mais numa máquina mais lenta).

### Onde está a análise

Em `fall-classifier-bench/` (rodar com um venv que tenha `numpy`, `pandas`,
`scikit-learn`, `onnxruntime`, `matplotlib` e `torch`; `compare_transition_errors.py`
importa o torch):

| Script | Para quê |
|---|---|
| `inventory_captures.py`, `inventory_math.py` | Conferir os dados e o orçamento de janelas antes de gerar qualquer métrica |
| `eval_captures.py` | Métricas dos dois GRUs (clipe e janela), sweep de `confirm_n`, McNemar |
| `eval_math_captures.py` | Modelo matemático nas configs de deploy (12 fps, janela 30) e reduzida (10 fps, janela 20) |
| `bench_latency_captures.py` | Latência de inferência |
| `analyze_new_takes.py`, `metrics_new_takes.py` | Análise com reamostragem por tempo dos takes gravados por este app, com gráficos |
| `full_report.py` | Relatório consolidado em Markdown |

Esses scripts importam de `../modelos/data_prep/` e `../modelos/training/gru/`, então o
repositório `modelos` precisa estar clonado ao lado. Alguns caminhos de dados estão fixos
para a máquina Windows original (`D:\...\captura pontos\data`); ajuste com `--data-dir`
ou editando `DEFAULT_DATA_DIR` em `captures_dataset.py`.

### Preferências de trabalho do usuário (Erik)

- Português do Brasil. Explicar o **porquê** antes de propor mudanças grandes.
- **Não fazer commit nem push sem aprovação explícita.**
- Não alterar arquivos de modelos/notebooks já em uso; criar arquivos novos ao lado.
- Decisões metodológicas precisam ser documentadas: o PFC será defendido perante banca.
- Ser honesto sobre n pequeno. Com 1–2 takes por classe, métricas são plausibilidade, não estimativa.
