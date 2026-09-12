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
| `--conf` | `0.25` | Confiança mínima do YOLO para considerar uma detecção |
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
1. O app abre a câmera e carrega o YOLO pose (só uma vez).
2. Aparece o menu de cenários, agrupado, com o próximo número de take de cada um.
3. Você digita o número do cenário e aperta ENTER.
4. Uma contagem regressiva de --countdown segundos te dá tempo de se posicionar.
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
Gravando 'sentando' em 1...
[janela de captura abre — pressione SPACE ao terminar a tomada]
Salvo: 1 clipe(s) -> ['sentando_take01_p0']

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
│       └── <scenario_id>_take01_p0_labels.npy   # (T,) — 0=ADL ou 2=FALL, uniforme na tomada
└── manifest.csv
```

`manifest.csv` tem uma linha por clipe gravado:

| Coluna | O que é |
|---|---|
| `clip_id` | Identificador único do clipe (`<scenario_id>_take<NN>_p<i>`) |
| `source` | Constante `"keypoint_extraction"` — identifica a origem do dado |
| `is_fall` | `True`/`False`, derivado do `label` do cenário |
| `src_fps` | FPS da câmera no momento da gravação |
| `px_path` / `lbl_path` | Caminhos absolutos dos `.npy` correspondentes |
| `group_id` | `<scenario_id>_take<NN>` — mesma tomada = mesmo group_id, útil para não misturar pessoas do mesmo take entre splits de treino/validação no futuro |
| `scenario_id` / `scenario_label` | Igual ao `id`/`label` de `scenarios.yaml` |
| `take_number` | Número da tomada dentro do cenário |
| `person_idx` | Índice da pessoa dentro da tomada (`0`, `1`, ...) — ver seção seguinte |
| `n_frames` | Quantos frames a tomada teve |
| `session_id` | Identifica a execução do app em que o clipe foi gravado |
| `recorded_at` | Timestamp UTC |
| `operator_note` | Reservado para anotação manual futura (hoje sempre vazio) |

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
