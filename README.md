# Price Tag Pipeline

**Модели:** https://drive.google.com/file/d/1-aiahzHSmZEMWJwmHQE2XroKYFACY0iK/view?usp=drive_link

Проект для детекции, трекинга и подсчета уникальных магазинных ценников на видео, где камера движется вдоль полки, а ценники в нормализованном кадре проходят слева направо.

Текущий pipeline:

- читает видео;
- нормализует ориентацию кадра;
- находит ценники через детектор;
- связывает детекции в треки;
- считает уникальные стабильные ценники;
- сохраняет CSV, кропы ценников и диагностическое overlay-видео;
- опционально отправляет выбранные кропы в OpenAI-compatible OCR/VLM сервис и формирует `structured_price_tags.csv`.

OCR и извлечение текста с ценника выделены в отдельный модуль и могут работать как с локальным LM Studio, так и с llama.cpp/VLM сервисом в Docker или на удаленной машине.

## Структура проекта

```text
price_tag_pipeline/
  run_pipeline.py          # основной запуск пайплайна
  api/                     # FastAPI backend для frontend-загрузки видео
  core/                    # CLI, orchestration, вывод, геометрия, визуализация
  detector/                # ONNX YOLO и эвристический детектор
  tracker/                 # треки, наблюдения, matching, правила подсчета
  ocr/                     # точка расширения под OCR
  vlm/                     # внешний VLM-сервис для структурного чтения ценника
  codes/                   # точка расширения под barcode и QR-code
  sources/                 # источники кадров, сейчас видео
  docs/
    architecture.md        # обзор архитектуры и контрактов

tools/
  track_price_tags.py      # backward-compatible wrapper для старого запуска

frontend/
  src/                     # React/Vite интерфейс загрузки видео и скачивания CSV

models/
  openfoodfacts-price-tag-detection/
    weights/
      model_ir_8_opset_17.onnx
  lenta_quality_cnn.npz
  qr_detector/
```

## Системные требования

- Python 3.11+.
- macOS или Linux. На Windows лучше запускать через WSL2.
- 8+ GB RAM для CLI без OCR; для VLM/OCR требования зависят от модели LM Studio или llama.cpp.
- Node.js 20+ и npm, если нужен frontend.
- Docker и Docker Compose, если нужен контейнерный запуск pipeline или локальный llama.cpp-сервис.
- Локальный OpenAI-compatible VLM сервис нужен только для `--run-ocr`: LM Studio, llama.cpp server или удаленный совместимый endpoint.
- Для наиболее качественного OCR рекомендуется VLM `NVIDIA/Nemotron-3-Nano 30B A3B` с квантованием не ниже 4-bit/Q4. Квантования 3-bit и ниже лучше оставлять только для smoke-test и отладки.

Установка Python-зависимостей из корня проекта:

```bash
python3 -m pip install -e .
```

Для frontend:

```bash
cd frontend
npm install
```

## Модели

Модели должны лежать в `models/` относительно корня репозитория. Архив с моделями: https://drive.google.com/file/d/1-aiahzHSmZEMWJwmHQE2XroKYFACY0iK/view?usp=drive_link

Ожидаемые пути по умолчанию:

```text
models/openfoodfacts-price-tag-detection/weights/model_ir_8_opset_17.onnx
models/lenta_quality_cnn.npz
models/qr_detector/
```

Каталог `models/` версионируется в репозитории, но веса можно переопределять локальными путями через CLI-параметры.

## Запуск

### 1. Простой CLI-запуск

Основной рабочий сценарий: задаем видео, папку результата и ONNX-детектор; остальные параметры остаются дефолтными.

```bash
python -m price_tag_pipeline.run_pipeline \
  /path/to/video.mp4 \
  --out-dir runs/video_run \
  --detector onnx
```

Если нужен полный CSV с OCR/VLM, включите `--run-ocr`. По умолчанию OCR смотрит в LM Studio на `http://localhost:1234/v1` и использует модель `local-vlm`.

```bash
python -m price_tag_pipeline.run_pipeline \
  /path/to/video.mp4 \
  --out-dir runs/video_ocr \
  --detector onnx \
  --run-ocr
```

Для стабильного качества распознавания ценников в качестве VLM рекомендуется `NVIDIA/Nemotron-3-Nano 30B A3B` с квантованием не ниже 4-bit/Q4.

Старый entrypoint тоже работает:

```bash
python tools/track_price_tags.py \
  /path/to/video.mp4 \
  --out-dir runs/video_run \
  --detector onnx
```

### 2. Запуск с frontend

Frontend отправляет видео в FastAPI backend, получает `job_id`, опрашивает статус обработки и затем скачивает CSV. По умолчанию UI вызывает backend через same-origin путь `/api/video/jobs`; в dev-режиме Vite проксирует `/api` на `http://localhost:8020`.

Запустите backend:

```bash
PRICE_TAG_API_PIPELINE_ARGS="--detector onnx --run-ocr" \
uvicorn price_tag_pipeline.api.main:app --host 0.0.0.0 --port 8020
```

Если OCR/VLM пока не нужен, уберите `--run-ocr` из `PRICE_TAG_API_PIPELINE_ARGS`.

В другом терминале запустите frontend:

```bash
cd frontend
npm run dev
```

Откройте `http://localhost:8021`, загрузите видео и дождитесь CSV. Для доступа с другого компьютера используйте IP или домен сервера: `http://<server-ip>:8021`. Настройки pipeline для backend передаются через `PRICE_TAG_API_PIPELINE_ARGS`, а параметры OCR можно задавать через env-переменные `PRICE_TAG_OCR_BASE_URL`, `PRICE_TAG_OCR_MODEL`, `PRICE_TAG_OCR_API_KEY` и другие.

Production-вариант через nginx:

```bash
docker compose up --build frontend
```

После запуска UI доступен на `http://<server-ip>:8021/` и `https://<server-ip>:8443/`, а backend закрыт за nginx-прокси `/api`. Порты можно переопределить через `FRONTEND_HTTP_PORT` и `FRONTEND_HTTPS_PORT`, например `FRONTEND_HTTP_PORT=80 FRONTEND_HTTPS_PORT=443 docker compose up --build frontend`, если эти порты свободны или обслуживаются внешним reverse proxy.

### 3. Расширенный запуск и Docker

Пример быстрого smoke-test без записи overlay-видео:

```bash
python -m price_tag_pipeline.run_pipeline \
  /path/to/video.mp4 \
  --out-dir runs/smoke_onnx \
  --detector onnx \
  --frame-step 120 \
  --no-overlay-video \
  --min-track-frames 1
```

Пример полного локального запуска с явными OCR и tracking-параметрами:

```bash
python -m price_tag_pipeline.run_pipeline \
  /path/to/video.mp4 \
  --out-dir runs/full_ocr \
  --detector onnx \
  --frame-step 3 \
  --run-ocr \
  --ocr-base-url http://localhost:1234/v1 \
  --ocr-model local-vlm
```

Docker-секция ниже относится к CLI-пайплайну; FastAPI backend и frontend запускаются локально командами из предыдущего блока.

Сборка:

```bash
docker compose build price-tag-pipeline
```

Запуск с локальным LM Studio на хосте:

```bash
PRICE_TAG_DATA_DIR=/path/to/video_dir \
PRICE_TAG_OCR_BASE_URL=http://host.docker.internal:1234/v1 \
docker compose run --rm price-tag-pipeline \
  /data/video.mp4 \
  --out-dir /app/runs/video_run \
  --detector onnx \
  --frame-step 3 \
  --run-ocr
```

Запуск с llama.cpp в этом же `docker-compose`:

```bash
docker compose up --build

PRICE_TAG_DATA_DIR=/path/to/video_dir \
PRICE_TAG_OCR_BASE_URL=http://llama-cpp:8080/v1 \
docker compose run --rm price-tag-pipeline \
  /data/video.mp4 \
  --out-dir /app/runs/video_run \
  --detector onnx \
  --frame-step 3 \
  --run-ocr
```

В обоих вариантах prompt, JSON-схема, отключение reasoning и параметры OCR берутся из локального модуля `price_tag_pipeline/ocr/extractor.py` и CLI/env-настроек пайплайна. Модели детектора и quality-классификатора нужно положить в локальную папку `models/`; она монтируется в контейнер как `/app/models`.

## Детекторы

Доступны два режима:

- `--detector onnx` - основной режим, YOLO11x / Ultralytics ONNX-модель для детекции ценников.
- `--detector heuristic` - старый эвристический fallback по красным областям ценника, полезен для быстрых smoke-test и диагностики без модели.

Модель по умолчанию:

```text
models/openfoodfacts-price-tag-detection/weights/model_ir_8_opset_17.onnx
```

Каталог `models/` предназначен для весов и экспортов, которые нужны pipeline по умолчанию. Если нужно протестировать альтернативный детектор, используйте `--onnx-model` и не меняйте дефолтные веса без отдельного решения.

Переопределить модель можно так:

```bash
--onnx-model /path/to/model.onnx
```

## Основные параметры

```text
--width                     ширина кадра после нормализации ориентации
--frame-step                обрабатывать каждый N-й исходный кадр
--detector                  heuristic или onnx
--count-mode                seen или line
--min-track-frames          минимальная длина валидного трека
--max-track-height          опциональный cap высоты bbox; 0 отключает фильтр, это дефолт
--max-link-distance         максимальная дистанция matching между треком и детекцией
--max-backward-step         допустимое движение назад по X
--max-y-jump                допустимый скачок по Y
--right-edge-retire-ratio   граница справа, где трек принудительно завершается
--right-edge-ignore-ratio   зона справа, где новые детекции не участвуют в matching
--max-crop-candidates       сколько лучших crop-кандидатов хранить на каждый трек, по умолчанию 24
--deskew-crops              выпрямлять кропы аффинным преобразованием
--no-deskew-crops           отключить выпрямление кропов
--crop-deskew-max-angle     максимальный допустимый угол выпрямления
--crop-deskew-min-angle     минимальный угол, ниже которого поворот не применяется
--crop-deskew-min-confidence минимальная уверенность оценки угла
--quality-model             ONNX или NPZ-модель бинарного классификатора качества кропа
--no-quality-model          отключить quality-классификатор
--quality-input-size        размер входа quality-модели
--quality-input-mode        grayscale или rgb, по умолчанию grayscale
--quality-threshold         порог принятия pass-класса
--quality-pass-class        класс хорошего кропа, по умолчанию 1
--quality-normalization     none или imagenet
--run-ocr                   отправить сохраненные кропы в локальную OCR/VLM модель
--ocr-base-url              OpenAI-compatible URL LM Studio, по умолчанию http://localhost:1234/v1
--ocr-model                 model id или alias, по умолчанию local-vlm
--ocr-max-tokens            лимит генерации OCR-модели, по умолчанию 900
--ocr-disable-reasoning     отключить reasoning для OCR-модели, это дефолт
--ocr-enable-reasoning      включить reasoning обратно для OCR-модели
--ocr-image-preprocess      none, grayscale, clahe или vlm-board; по умолчанию none
--ocr-include-quality-failed дополнительно отправлять в OCR fallback-кропы треков, не прошедших quality; это дефолт
--no-ocr-include-quality-failed OCR только по quality-passed кропам
--ocr-limit                 обработать только первые N кропов, 0 значит все
--code-detector             включить NCNN-детектор QR/barcode зон внутри OCR-кропа
--no-code-detector          отключить NCNN-детектор и вернуться к scan всего OCR-кропа
--code-detector-dir         папка NCNN-модели, по умолчанию models/qr_detector
--code-detector-conf-threshold confidence threshold для QR/barcode region detector
--code-detector-padding-ratio дополнительный контекст вокруг QR/barcode bbox, по умолчанию 0.50
--code-detector-resize-mode stretch или letterbox, по умолчанию stretch
--code-detector-class-roles соответствие классов NCNN к decoder-ролям, по умолчанию 0=qr,1=barcode,2=matrix
--no-code-detector-fallback-full-crop не запускать старый full-crop scan, если region detector ничего не прочитал
--no-overlay-video          не писать counting_overlay.mp4
--debug-frames              список кадров для сохранения overlay JPG
```

По умолчанию используется `--count-mode seen`: считаются все стабильные уникальные треки в кадрах, без линии пересечения.

Режим `--count-mode line` оставлен для экспериментов, где нужно считать только проход через вертикальную линию.

## Выходные артефакты

В `--out-dir` создаются:

```text
report.json                общий отчет по запуску
unique_price_tags.csv      уникальные посчитанные ценники
all_valid_tracks.csv       все валидные треки
track_history.csv          история bbox каждого трека по кадрам
frame_counts.csv           статистика по каждому обработанному кадру
detections.csv             сырые детекции по кадрам
structured_price_tags.csv  финальный структурированный CSV после OCR, если включен --run-ocr
ocr_raw_responses.jsonl    сырые OCR-ответы и QR debug, если включен --run-ocr
code_crops/                QR/barcode зоны, найденные NCNN-детектором внутри кропов ценников
counting_overlay.mp4       видео с bbox, id треков и счетчиками
crops/                     сохраненные финальные кропы; при включенном quality только quality_passed=1
overlays/                  debug-кадры с разметкой
```

Главный файл для контроля результата сейчас:

```text
unique_price_tags.csv
```

Он содержит `track_id`, кадры появления, координаты лучшего bbox, качество кропа и путь к изображению ценника. Поля `best_full_*` остаются в координатах нормализованного полного кадра, а `source_x_min/source_y_min/source_x_max/source_y_max` отдают bbox в пикселях исходного видео.

## Текущая логика подсчета

1. Видео нормализуется по ориентации, для текущих роликов по умолчанию используется поворот против часовой стрелки.
2. Детектор возвращает bbox ценников на нормализованном кадре.
3. Трекер связывает bbox между кадрами с ограничениями по расстоянию, вертикальному скачку и движению назад.
4. Для каждого трека хранится несколько лучших crop-кандидатов, а не только один кадр.
5. Crop-кандидаты выпрямляются аффинным преобразованием, если удалось оценить наклон.
6. Если подключена quality-модель, кандидаты проверяются бинарным классификатором качества на ч/б входе и перебираются до первого кропа с pass-классом `1`.
7. Треки, дошедшие до правого края, завершаются, чтобы не перехватывать соседние ценники.
8. Валидным считается трек с достаточным числом наблюдений, приемлемой геометрией и признаками ценника.
9. В режиме `seen` каждый валидный трек считается уникальным ценником.
10. Если включен `--run-ocr`, выбранные обработанные кропы передаются в локальную OpenAI-compatible OCR/VLM модель LM Studio по одному изображению. По умолчанию OCR также получает fallback-кропы для треков, где quality не пропустил ни один кандидат, чтобы финальный CSV сохранял строку на каждый уникальный трек.
11. После VLM по тем же OCR-кропам запускается NCNN-детектор QR/barcode зон, найденные зоны сохраняются в `code_crops/` и передаются в общий `codes`-reader.
12. Если region detector ничего не прочитал, по умолчанию выполняется старый fallback scan всего OCR-кропа.
13. После объединения трекинга, OCR, QR и barcode пишется `structured_price_tags.csv`.

## Выпрямление кропов

Перед сохранением финального кропа пайплайн пытается выровнять ценник относительно его наклона. Это аффинный поворот кропа, bbox-трекинг при этом не меняется.

Оценка угла идет двумя способами:

1. По красной области ценника через `red_mask` и `minAreaRect`.
2. Fallback по длинным почти горизонтальным линиям через Hough.

Если уверенности недостаточно или угол слишком мал, кроп остается исходным.

По умолчанию deskew включен. Отключить можно так:

```bash
--no-deskew-crops
```

В `unique_price_tags.csv` добавляются поля:

```text
deskew_applied
deskew_angle
deskew_confidence
deskew_method
```

## Выбор кропа через классификатор качества

Бинарный классификатор качества подключается через ONNX или локальную `.npz` CNN-модель.

Локальная модель по умолчанию:

```text
models/lenta_quality_cnn.npz
```

Если этот файл существует, пайплайн подключает его автоматически. Отключить quality-проверку можно так:

```bash
--no-quality-model
```

Для `.npz`-модели `image_size`, `input_mode`, `downsample_factor` и `decision_threshold` читаются из самой модели. Текущая модель использует grayscale-вход и threshold `0.38`.

Запуск с дефолтной `.npz`-моделью:

```bash
python -m price_tag_pipeline.run_pipeline \
  /Users/alexeyguchko/lenta/Данные/43_15/43_15.mp4 \
  --out-dir /Users/alexeyguchko/lenta/Данные/43_15/price_tag_tracking_quality \
  --width 720 \
  --detector onnx \
  --frame-step 3
```

Для ручного указания другой модели:

```bash
--quality-model /path/to/quality_classifier.onnx
```

Логика выбора:

1. Трекер копит до `--max-crop-candidates` лучших кадров одного и того же `track_id`; дефолт `24` дает quality-модели больше шансов найти несмазанный кадр без изменения подсчета треков.
2. На финальном этапе кандидаты сортируются по старому `crop_quality`.
3. Для quality-модели каждый кандидат преобразуется в ч/б tensor. Сам сохраненный кроп при этом не преобразуется.
4. Quality-модель проверяет кандидаты по очереди.
5. Первый кандидат с class `1` становится финальным кропом и сохраняется в `crops/`.
6. Если все кандидаты получили class `0`, лучший fallback-кандидат остается в CSV для диагностики, но файл в `crops/` не сохраняется, `crop_saved=0`, `crop_path` пустой.

В `unique_price_tags.csv` добавлены поля:

```text
selected_crop_rank
quality_label
quality_score
quality_passed
quality_attempts
rejected_quality_candidates
crop_candidates
crop_saved
```

## LM Studio OCR

После выбора финальных кропов слой OCR может отправить каждое изображение в локальный OpenAI-compatible сервер LM Studio. OCR запускается только при явном флаге:

```bash
--run-ocr
```

Основные параметры:

```text
--ocr-base-url http://localhost:1234/v1
--ocr-model local-vlm
--ocr-max-tokens 900
--ocr-timeout 180
--ocr-retries 1
--ocr-response-format
--ocr-image-preprocess none
```

Также поддержаны переменные окружения `PRICE_TAG_OCR_*` и совместимые `LLAMA_CPP_*`, например `PRICE_TAG_OCR_BASE_URL`, `PRICE_TAG_OCR_MODEL`.
Для Nemotron в LM Studio reasoning выключен через local `model.yaml` (`enableThinking.defaultValue=false`); pipeline дополнительно отправляет `/no_think` и `chat_template_kwargs.enable_thinking=false`.
Для A/B-отладки перед отправкой в VLM можно включить `--ocr-image-preprocess vlm-board`: модель получает один composite-кадр из полного цветного ценника и увеличенных контрастных нижних фрагментов. Также доступны `grayscale` и `clahe`. По умолчанию используется `none`, потому что на текущей VLM composite-режим может улучшать мелкие поля, но иногда портит основные значения.

OCR-парсер ожидает JSON со строго заданными ключами:

```text
product_name
price_default
price_card
price_discount
barcode
discount_amount
id_sku
print_datetime
code
additional_info
color
special_symbols
wholesale_level_1_count
wholesale_level_1_price
wholesale_level_2_count
wholesale_level_2_price
```

Prompt ориентирован на зоны, показанные в материалах по расшифровке ценников: `price_default` под QR-кодом и над ценой по карте, `id_sku` под процентной плашкой или в нижней зоне, `print_datetime`, цифры под линейным штрихкодом, символ выкладки `Ш`/`Л`/`К`, доп. инфо вроде `номер на весах` или `Удачная упаковка`, а также явные оптовые пороги `от N шт`. Поле `code` заполняется только если виден отдельный код выкладки; цену без карты и `id_sku` туда писать нельзя.

Перед записью `structured_price_tags.csv` значения проходят нормализацию в `price_tag_pipeline/ocr/schema.py`. Это защищает итоговый CSV от нестрогих ответов VLM:

- основные цены приводятся к формату `123,45`;
- QR-цены приводятся к формату `123.45`;
- barcode и SKU очищаются до цифр;
- цены вида `55 39` и `1 284 29` трактуются как `55,39` и `1284,29`;
- `discount_amount` приводится к виду `-23%` или `-50 руб`;
- `special_symbols` нормализуется к `Ш`, `Л`, `К` или `нет`;
- если VLM ошибочно положила цену без карты в `code`, а `price_default` пустой, значение переносится в `price_default`, а `code` очищается;
- timestamp и целочисленные поля приводятся к integer-строкам;
- координаты пишутся как числовые строки, для дробных значений используется запятая;
- отсутствующие optional/QR значения пишутся как `нет`, где такой формат используется в эталонном CSV.

Если QR не прочитан, CSV-конструктор аккуратно заполняет часть QR-совместимых полей из видимых OCR-значений: `qr_code_barcode` из напечатанного barcode, `price1_qr` из цены без карты, `price4_qr` из цены по карте, а `wholesale_level_*` из явно прочитанных оптовых порогов. Финальный CSV сохраняет совместимость с текущей эталонной разметкой, включая заголовок `wholesale_level_1_coun`.

Точка интеграции:

```text
price_tag_pipeline/ocr/extractor.py
price_tag_pipeline/ocr/stage.py
```

Финальный результат пишется в:

```text
structured_price_tags.csv
ocr_raw_responses.jsonl
```

## Barcode и QR-code

Логика поиска barcode и QR-code выделена отдельно от детектора ценников и OCR:

```text
price_tag_pipeline/codes/
```

После VLM на тех же кропах запускается NCNN-детектор из `models/qr_detector`. Он находит зоны QR/barcode внутри ценника, сохраняет подготовленные region crops в `code_crops/`, и уже эти изображения передаются в `codes/reader.py`. По умолчанию классы интерпретируются как `0=qr,1=barcode,2=matrix`; `matrix` обрабатывается как 2D-код и маппится в те же QR-поля. Если детектор недоступен или не дал читаемый результат, по умолчанию сохраняется старый fallback: scan всего OCR-кропа.

Reader читает QR через `pyzbar`/`zxingcpp`, если они доступны, и OpenCV `QRCodeDetector`; для линейных barcode дополнительно используется OpenCV `barcode_BarcodeDetector`. В `structured_price_tags.csv` попадают поля:

```text
qr_code_barcode
price1_qr
price2_qr
price3_qr
price4_qr
wholesale_level_1_coun
wholesale_level_1_price
wholesale_level_2_count
wholesale_level_2_price
action_price_qr
action_code_qr
```

## Сбор кропов для классификатора качества

Для обучения бинарного классификатора качества кропов есть отдельная standalone-утилита:

```text
tools/collect_price_tag_crops.py
```

Она не импортирует основной пайплайн. Скрипт сканирует `unique_price_tags.csv` и папки `crops/`, копирует изображения в одну папку и пишет `manifest.csv` с исходными путями и metadata треков.

Пример сборки всех найденных кропов из папки с экспериментами:

```bash
python tools/collect_price_tag_crops.py \
  /Users/alexeyguchko/lenta/Данные/43_15 \
  --out-dir /Users/alexeyguchko/lenta/Данные/43_15/quality_classifier_crops \
  --class-name unlabeled
```

По умолчанию включен dedupe по содержимому изображения:

```text
--dedupe hash
```

Если нужны все файлы без удаления дублей:

```bash
--dedupe none
```

После копирования можно вручную разложить изображения из `unlabeled/` на два класса, например `normal/` и `bad/`, оставив `manifest.csv` как связь с исходными треками.

## Документация

Архитектура и поток данных описаны в:

```text
price_tag_pipeline/docs/architecture.md
```
