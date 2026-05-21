# MultiModalHugs — Project Notes (feature/siamese-ot-av)

## Objetivo del proyecto

Sign Language Translation (SLT) de LSC (Llengua de Signes Catalana).
El modelo de vídeo solo tiene mal rendimiento. La idea es usar el audio (Catalan speech)
como señal auxiliar durante el preentrenamiento para mejorar el encoder de vídeo,
y después hacer fine-tuning solo con vídeo para SLT.

**Dataset:** LSC-Parlament — interpretación parlamentaria.
El mismo `.mp4` contiene ambas pistas: el hablante habla en catalán y el intérprete lo
signa en LSC. El audio **siempre precede** al vídeo (el intérprete escucha antes de signar).

---

## Arquitectura base (MultiModalEmbedderModel)

```
input_frames → FeatureExtractor → MultimodalMapper → merge_modalities → Backbone → logits
                  (CLIP)            (Linear/CNN)        (encoder_prompt)   (mBART/M2M)
```

- **FeatureExtractor**: wraps cualquier modelo HF. Tipos soportados: `"clip"`, `"whisper"`.
- **MultimodalMapper**: proyecta de `feat_dim` a `d_model`. Tipos: `linear`, `adapter`, `cnn_adapter`.
- **merge_modalities**: prepend del `encoder_prompt`, inserta EOS. Se llama en cada forward.
- **Backbone**: mBART / M2M-100. El decoder genera la traducción autoregressivamente.
- **Convención de máscara**: 0 = padding, 1 = válido en todo el framework.

---

## Nuevos ficheros añadidos (branch feature/siamese-ot-av)

### Dataset
**`multimodalhugs/data/datasets/videoaudio2text.py`**
- `VideoAudio2TextDataset` + `VideoAudio2TextDataConfig`
- Lee TSVs de LSC-Parlament con columnas: `signal`, `signal_start`, `signal_end`, `audio_start`, `audio_end`, `encoder_prompt`, `decoder_prompt`, `output`.
- Emite **dos columnas distintas** para el mismo path mp4: `video_signal` y `audio_signal`.
  Esto evita la colisión de `primary_field` en los ProcessorSlots (el primero sobrescribiría
  el valor de la columna `signal` con un tensor antes de que el segundo slot lo pueda leer).

### Módulo OT
**`multimodalhugs/modules/sinkhorn.py`**
- `sinkhorn_loss(x, y, x_mask, y_mask, epsilon, max_iter)`: OT entre dos conjuntos de vectores de un sample.
- `batch_sinkhorn_loss(x, y, x_mask, y_mask, epsilon, max_iter)`: media sobre el batch.
- Implementación en dominio logarítmico (numéricamente estable para epsilon pequeño).
- **Matriz de coste**: distancia coseno (`1 - x @ y.T` sobre vectores normalizados a esfera unitaria).
- **Funciona con batch_size = 1**: el OT se calcula *dentro* de un sample (T_video frames vs T_audio frames),
  no entre samples del batch. Por eso es compatible con batch_size = 1 en el cluster.

### Modelo Siamese
**`multimodalhugs/models/siamese_multimodal_embedder/`**

`configuration_siamese_multimodal_embedder.py` — extiende `MultiModalEmbedderConfig` con:
- `audio_feature_extractor_type` (e.g. `"whisper"`)
- `audio_pretrained_feature_extractor` (e.g. `"openai/whisper-medium"`)
- `audio_feat_dim` (1024 para Whisper-medium)
- `audio_freeze_feature_extractor` (**debe ser `true`** — ver decisiones de diseño)
- `audio_multimodal_mapper_type`, `_layer_norm_before`, `_layer_norm`, `_activation`, `_dropout`, `_factor`
- `audio_freeze_multimodal_mapper`
- `ot_lambda`, `sinkhorn_epsilon`, `sinkhorn_max_iter`

`modeling_siamese_multimodal_embedder.py` — extiende `MultiModalEmbedderModel`:
- `forward(input_audio=None)`: si `input_audio=None`, delega **exactamente** al padre (modo vídeo puro).
- `forward(input_audio=tensor)`:
  1. Vídeo: `FeatureExtractor` → `MultimodalMapper` → `video_repr [B, T_v, D]`
  2. Audio: `AudioFeatureExtractor` (frozen) → `AudioMultimodalMapper` → `audio_repr [B, T_a, D]`
  3. OT loss: `batch_sinkhorn_loss(video_repr, audio_repr, ...)`
  4. Vídeo continúa: `merge_modalities` → `Backbone` → MT loss
  5. `total_loss = MT_loss + ot_lambda * OT_loss`
- `prepare_inputs_for_generation`: elimina `input_audio` antes de llamar a `generate()`.
  **La generación es siempre modo vídeo puro**, incluso si el batch tiene audio.

### Procesador de audio — soporte mp4
**`multimodalhugs/processors/speech_modality_processor.py`** (modificado)
- `_load_waveform()` ahora despacha según extensión del fichero:
  - `.mp4`, `.mkv`, `.mov`, etc. → `_load_waveform_from_mp4()` usando **PyAV**
    (más preciso para audio comprimido AAC dentro de contenedores de vídeo)
  - `.wav`, `.flac`, etc. → torchaudio (comportamiento original, sin cambios)
- **Backward compatible**: los pipelines `speech2text` existentes no cambian nada.

---

## Decisiones de diseño clave

### Por qué Optimal Transport (no contrastivo/InfoNCE)
- El cluster limita a **batch_size = 1**. Los métodos contrastivos (InfoNCE) necesitan
  negativos de otros samples del mismo batch → no funcionan con BS=1.
- OT es una pérdida *per-sample*: compara las T_video representaciones contra las T_audio
  representaciones dentro de un solo ejemplo. Funciona con cualquier BS ≥ 1.
- El audio **siempre precede** al vídeo (lag del intérprete). OT es invariante a posición
  (encuentra el plan de transporte óptimo entre distribuciones), así que el desfase
  temporal no es un problema.

### Por qué congelar Whisper (`audio_freeze_feature_extractor: true`)
- Whisper es un teacher con representaciones fonéticas ricas y estables.
- Si ambos encoders son libres, el OT puede minimizarse haciendo que ambos converjan
  hacia representaciones triviales (colapso de modo) en lugar de que el encoder de vídeo
  aprenda características fonéticas reales.
- Congelar Whisper ancla el espacio objetivo y ahorra ~300M params de gradientes/optimizer state.
- **El audio mapper sí debe ser trainable** (`audio_freeze_multimodal_mapper: false`):
  es el puente ligero de 1024-d → d_model y necesita aprender la proyección.

### Colisión de primary_field en ProcessorSlots
Si dos slots leen la misma columna TSV (`signal`), el primer slot la reemplaza por un tensor
antes de que el segundo slot pueda leerla. Solución: `VideoAudio2TextDataset` emite
`video_signal` y `audio_signal` con el mismo path pero keys distintos. Cada slot mapea
su propia key: `{"video_signal": "signal", ...}` y `{"audio_signal": "signal", ...}`.

---

## Estructura de ficheros relevante

```
multimodalhugs/
├── data/
│   ├── __init__.py                          (+ lazy load VideoAudio2TextDataset)
│   └── datasets/
│       ├── video2text.py                    (existente)
│       ├── speech2text.py                   (existente)
│       └── videoaudio2text.py               (NUEVO)
├── modules/
│   ├── __init__.py                          (+ exporta sinkhorn_loss, batch_sinkhorn_loss)
│   └── sinkhorn.py                          (NUEVO)
├── models/
│   ├── multimodal_embedder/                 (existente — no modificado)
│   └── siamese_multimodal_embedder/         (NUEVO)
│       ├── __init__.py
│       ├── configuration_siamese_multimodal_embedder.py
│       └── modeling_siamese_multimodal_embedder.py
├── processors/
│   └── speech_modality_processor.py         (modificado: soporte mp4 via PyAV)
└── training_setup/
    └── general_training_setup.py            (+ entrada "videoaudio2text" en _DATASET_IMPORT_MAP)
```

---

## YAML de entrenamiento (Fase 1 — Siamese pretraining)

```yaml
model:
  type: siamese_multimodal_embedder
  # Vídeo
  feature_extractor_type: clip
  pretrained_feature_extractor: openai/clip-vit-base-patch32
  multimodal_mapper_type: linear
  multimodal_mapper_layer_norm_before: true
  multimodal_mapper_layer_norm: false
  multimodal_mapper_activation: false
  multimodal_mapper_dropout: 0.1
  feat_dim: 512
  freeze_feature_extractor: false
  freeze_multimodal_mapper: false
  # Audio (Whisper como teacher fijo)
  audio_feature_extractor_type: whisper
  audio_pretrained_feature_extractor: openai/whisper-medium
  audio_multimodal_mapper_type: linear
  audio_multimodal_mapper_layer_norm_before: true
  audio_multimodal_mapper_layer_norm: false
  audio_multimodal_mapper_activation: false
  audio_multimodal_mapper_dropout: 0.1
  audio_feat_dim: 1024
  audio_freeze_feature_extractor: true   # <-- IMPORTANTE: congelar Whisper
  audio_freeze_multimodal_mapper: false
  # OT
  ot_lambda: 1.0
  sinkhorn_epsilon: 0.1
  sinkhorn_max_iter: 100
  # Backbone
  backbone_type: m2m_100
  pretrained_backbone: facebook/m2m100_418M
  freeze_backbone: false
  freeze_decoder_embed_tokens: false
  freeze_encoder_embed_tokens: false
  freeze_lm_head: false
  max_length: 100

setup:
  seed: 3435
  output_dir: /home/usuaris.new/adria.capdevila.zurita/lsc-parlament/setup_multimodal/

training:
  run_name: parlament_siamese_ot
  output_dir: /home/usuaris.new/adria.capdevila.zurita/lsc-parlament/checkpoints_multimodal
  logging_dir: /home/usuaris.new/adria.capdevila.zurita/lsc-parlament/checkpoints_multimodal
  do_train: true
  do_eval: true
  predict_with_generate: true
  overwrite_output_dir: true
  eval_strategy: steps
  save_strategy: steps
  eval_steps: 500
  save_steps: 500
  logging_steps: 100
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 64
  learning_rate: 5e-05
  weight_decay: 0
  adam_beta1: 0.9
  adam_beta2: 0.998
  max_grad_norm: 0.0
  num_train_epochs: 1
  max_steps: 200000
  lr_scheduler_type: inverse_sqrt
  warmup_steps: 8000
  save_total_limit: 2
  seed: 3435
  dataloader_drop_last: false
  metric_for_best_model: sacrebleu
  metric_name: sacrebleu,chrf
  greater_is_better: true
  load_best_model_at_end: true
  remove_unused_columns: false
  dataloader_num_workers: 10
  dataloader_prefetch_factor: 4
  early_stopping_patience: 5
  fp16: true
  label_smoothing_factor: 0.1

data:
  dataset_type: videoaudio2text
  name: lsc_parlament_multimodal
  train_metadata_file: /home/usuaris.new/carlos.escolano/lsc-parlament/ca_filtered/train.tsv
  validation_metadata_file: /home/usuaris.new/carlos.escolano/lsc-parlament/ca_filtered/validation.tsv
  test_metadata_file: /home/usuaris.new/carlos.escolano/lsc-parlament/ca_filtered/test.tsv
  shuffle: true
  max_frames: 700

processor:
  slots:
    - processor_class: VideoModalityProcessor
      output_data_key: input_frames
      output_mask_key: attention_mask
      column_map:
        video_signal: signal
        signal_start: signal_start
        signal_end: signal_end
      processor_kwargs:
        custom_preprocessor_path: openai/clip-vit-base-patch32
        join_chw: false
        skip_frames_stride: 3

    - processor_class: SpeechModalityProcessor
      output_data_key: input_audio
      output_mask_key: audio_attention_mask
      column_map:
        audio_signal: signal
        audio_start: signal_start
        audio_end: signal_end
      processor_kwargs:
        custom_preprocessor_path: openai/whisper-medium

    - processor_class: TextModalityProcessor
      output_data_key: labels
      is_label: true
      column_map:
        decoder_prompt: target_prefix
        output: target
      processor_kwargs:
        tokenizer_path: facebook/m2m100_418M
        new_vocabulary: "__lsc__"
        role: target

    - processor_class: TextModalityProcessor
      output_data_key: encoder_prompt
      output_mask_key: encoder_prompt_length_padding_mask
      column_map:
        encoder_prompt: signal
      processor_kwargs:
        tokenizer_path: facebook/m2m100_418M
        new_vocabulary: "__lsc__"
        role: input

    - processor_class: TextModalityProcessor
      output_data_key: decoder_input_ids
      output_mask_key: decoder_attention_mask
      column_map:
        decoder_prompt: signal
      processor_kwargs:
        tokenizer_path: facebook/m2m100_418M
        new_vocabulary: "__lsc__"
        role: input
```

---

## YAML de evaluación solo vídeo (generate.py / Fase 2)

Para evaluar el modelo entrenado usando solo vídeo, usar `dataset_type: video2text`
con el mismo TSV — el dataset ignora las columnas de audio. El modelo siamese sin
`input_audio` en el batch entra automáticamente en modo vídeo puro.

```yaml
model:
  model_name_or_path: /ruta/al/checkpoint/siamese
  type: siamese_multimodal_embedder

data:
  dataset_type: video2text
  # mismo TSV que en entrenamiento
  ...

processor:
  pipeline: video2text
  tokenizer_path: facebook/m2m100_418M
  new_vocabulary: "__lsc__"
  modality_kwargs:
    custom_preprocessor_path: openai/clip-vit-base-patch32
    join_chw: false
    skip_frames_stride: 3
```

Durante `generate()`:
- `prepare_inputs_for_generation()` elimina `input_audio` si existiera.
- `forward(input_audio=None)` delega al padre: pipeline vídeo puro.
- Las métricas (sacrebleu/chrf) durante training también son vídeo puro (vía generate).
- La eval loss durante training sí incluye el OT loss (útil para monitorizar alineamiento).

---

## Notas de instalación en el cluster

```bash
git clone https://github.com/Adriacz/multimodalhugs.git
cd multimodalhugs
git checkout feature/siamese-ot-av
pip install -e ".[full]"
```

Dependencias clave: `torch<2.6`, `transformers<=4.44.2`, `av` (para leer audio de mp4).

---

## Estrategia de entrenamiento en dos fases (branch: feature/video-ft-checkpoint)

### Visión general

```
Fase 1 (HECHA):  speech2text  →  MultiModalEmbedderModel  →  checkpoint-best de audio
                                  Whisper → Mapper → M2M (backbone domain-adapted)

Fase 2 (nueva):  video2text   →  SiameseMultiModalEmbedderModel
                                  CLIP → VideoMapper → M2M (frozen)
                                  Whisper → AudioMapper (frozen, referencia Fase 1)
```

**Objetivo de Fase 2**: el backbone ya conoce el dominio LSC-Parlament gracias al audio.
Entrenar CLIP + VideoMapper para que sus representaciones se ajusten al espacio del backbone.
El audio branch frozen sirve de referencia para monitorizar si hay negative transfer.

---

### Nuevo flag: `train_with_audio`

`SiameseMultiModalEmbedderConfig.train_with_audio: bool = True`

Cuando `False`, en el `forward()` de training se ignora `input_audio` del batch aunque esté
presente → path vídeo puro, sin OT loss, sin gradiente en la rama de audio.
En eval (`self.training=False`) el flag no aplica, por lo que el dual eval del trainer
funciona normal: pasada de vídeo + pasada de audio.

**Por qué usar `videoaudio2text` aunque no se entrene con audio:**
Si se usara `video2text`, los batches de eval tampoco tendrían `input_audio` y la pasada
de audio eval quedaría sin datos. Con `videoaudio2text` los batches tienen ambas modalidades;
el modelo los ignora en training gracias a `train_with_audio=False`.

---

### Script de conversión de pesos

**`multimodalhugs/utils/convert_phase1_to_siamese.py`**
CLI: `mmhugs-convert-phase1` / `multimodalhugs-convert-phase1`

Renombra las keys del checkpoint de Fase 1:
- `feature_extractor.*` → `audio_feature_extractor.*`
- `multimodal_mapper.*` → `audio_mapper.*`
- `backbone.*` → `backbone.*` (sin cambios)

Inicializa CLIP y VideoMapper frescos (desde `openai/clip-vit-base-patch32`).
Guarda el checkpoint siamese con `freeze_backbone=True` (y embed_tokens/lm_head también)
forzado independientemente de lo que tuviera Fase 1.

**Notas sobre los pesos missing del backbone:**
Las keys `backbone.model.encoder.embed_tokens.weight`, `backbone.model.decoder.embed_tokens.weight`
y `backbone.lm_head.weight` aparecen como "missing" al hacer `load_state_dict`. Es esperado:
son pesos tied en M2M-100 (se guardan una sola vez en safetensors y se restauran
automáticamente por el mecanismo de tying del modelo).

```bash
mmhugs-convert-phase1 \
    --phase1_checkpoint /home/usuaris.new/adria.capdevila.zurita/lsc-parlament/checkpoints_speech_v5/train/checkpoint-best \
    --output_dir        /home/usuaris.new/adria.capdevila.zurita/lsc-parlament/phase2_siamese_init
```

---

### YAML de Fase 2

**`examples/multimodal_translation/lsc_parlament_video_ft/config_video_ft.yaml`**

Puntos clave respecto al YAML de Fase 1:
- `model.type: siamese_multimodal_embedder`
- `train_with_audio: false` — training vídeo puro
- `ot_lambda: 0.0` — sin OT
- `eval_audio: true` — dual eval habilitado
- `audio_freeze_feature_extractor: true` + `audio_freeze_multimodal_mapper: true` — rama audio completamente frozen
- `freeze_backbone: true` + `freeze_*_embed_tokens: true` + `freeze_lm_head: true` — backbone frozen
- `dataset_type: videoaudio2text` — necesario para que eval tenga `input_audio`
- `tokenizer_path` → apuntar al checkpoint-best de Fase 1

**Flujo completo de ejecución:**

```bash
git checkout feature/video-ft-checkpoint
pip install -e ".[full]"

# 1. Convertir checkpoint Fase 1
mmhugs-convert-phase1 \
    --phase1_checkpoint /path/to/phase1/checkpoint-best \
    --output_dir        /path/to/phase2/siamese_init

# 2. Setup dataset + processor (apuntar setup.output_dir al siamese_init)
mmhugs-setup --config_path examples/multimodal_translation/lsc_parlament_video_ft/config_video_ft.yaml \
             --output_dir /path/to/phase2/siamese_init

# 3. Entrenar
mmhugs-train --task translation \
             --config_path examples/multimodal_translation/lsc_parlament_video_ft/config_video_ft.yaml
```

**Métricas en eval:**
- `eval_sacrebleu` — BLEU del modelo de vídeo (lo que se está entrenando)
- `eval_audio_sacrebleu` — BLEU de la rama de audio frozen (referencia Fase 1)

Si `eval_sacrebleu` sube y `eval_audio_sacrebleu` se mantiene estable → no hay negative transfer.
