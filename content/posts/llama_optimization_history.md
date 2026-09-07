+++
title = 'Оптимизация llama.cpp для инференса на CPU+iGPU Intel Core Ultra 7 155H (Meteor Lake)'
description = 'Локальный инференс иногда нужен для решения моих задач. В рамках экспериментов добивался
максимальной доступной мне производительности на ноутбуке. Без дискретного GPU.'
date = 2026-09-07
draft = false
tags = ["AI", "HW"]
categories =  ["Искусственный Интеллект", "Инфраструктура", "Исследование"]
+++

**Железо:** Intel Core Ultra 7 155H (6P+8E+2LP, 22 потока), Intel Arc Graphics MTL (iGPU, unified memory), 64 GB RAM (доступно ~60 GB), Ubuntu 26.04 LTS.
**Репозиторий:** `~/ai/llama.cpp`, текущий коммит `e107984bc` (build 10788)
**Модель-эталон:** Qwen3-Coder-30B-A3B-Instruct Q4_K_M (MoE, 30.5B параметров / 3B активных, 17.28 GiB)

---

## TL;DR

- **Vulkan > SYCL** на Arc MTL: prefill 213 vs 95 t/s, decode 15.7 vs 18 t/s → Vulkan.
- **MoE 6× быстрее dense** на iGPU: 30B-A3B = 15.7 t/s decode, dense 27B = 2.5 t/s.
- **SYCL молча падал без `libumf.so.1`** в `LD_LIBRARY_PATH` — чинить нельзя оставить.
- **Итог**: локальный кодинг-стек llama-server (Vulkan, 32K) + kilo CLI, автозапуск через systemd.

---

## 1. Оптимизация сборки

### 1.1. Эволюция сборок

| Сборка            | Компилятор                   | Бэкенды                      | Дата       | Статус                        |
| ----------------- | ---------------------------- | ---------------------------- | ---------- | ----------------------------- |
| `build-native`    | GCC (системный)              | CPU (AVX2/FMA/BMI2) + Vulkan | 01.07.2026 | Рабочая, не трогать           |
| `build` (старая)  | GCC                          | CPU + Vulkan                 | до 04.09   | build 9722, все бенчи августа |
| `build` (текущая) | **icx/icpx** (oneAPI 2026.1) | CPU + **SYCL** + Vulkan      | 04.09.2026 | Основная, build 10788         |

### 1.2. Итоговый скрипт сборки `~/ai/llama.cpp/build.sh`

```bash
#!/bin/bash
set -e

# Инициализация окружения Intel oneAPI (|| true: setvars.sh выходит с кодом 3,
# если переменные уже установлены в текущей сессии — это не ошибка).
# ВАЖНО: не подключаем /opt/intel/openvino/setvars.sh - он экспортирует TBB_DIR
# на свою TBB 2021.13, а MKL-SYCL требует TBB 2023.1 (символ get_thread_reference_vertex).
source /opt/intel/oneapi/setvars.sh --force || true

# Полностью очищаем прошлую неудачную сборку
rm -rf build

# icx/icpx + GCC 15 toolchain (в gcc-13 нет libstdc++ headers, g++-13 не установлен)
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON \
  -DGGML_SYCL_F16=ON \
  -DGGML_VULKAN=ON \
  -DCMAKE_C_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15 -march=native -O3" \
  -DCMAKE_CXX_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15 -march=native -O3"

cmake --build build --config Release -j$(nproc)
```

### 1.3. Ключевые решения по сборке и почему

- **`icx/icpx` вместо GCC.** Компиляторы Intel генерируют код с лучшей векторизацией для SYCL-ядер; SYCL-бэкенд в принципе рассчитан на clang-семейство. CPU-часть при этом остаётся совместимой с ABI.
- **`--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15`.** icpx не включает собственные libstdc++ headers, а использует системный GCC toolchain. В системе GCC 15 — только он содержит нужные headers; попытка собрать с gcc-13 падала на отсутствии заголовков.
- **`-march=native -O3`.** Нативный код под Meteor Lake (AVX2/FMA/BMI2 и т.д.).
- **`GGML_SYCL=ON` + `GGML_VULKAN=ON` в одной сборке.** Оба бэкенда живут в одном бинарнике как отдельные shared-библиотеки (`libggml-sycl.so`, `libggml-vulkan.so`) — это позволило честно сравнивать их между собой одним `llama-bench --device`.
- **`GGML_SYCL_F16=ON`.** fp16-вычисления на iGPU: Arc MTL поддерживает fp16 нативно, это критично для скорости.
- **Конфликт TBB (грабли, задокументированные в скрипте).** OpenVINO-окружение экспортирует `TBB_DIR` на свою TBB 2021.13, а `libmkl_sycl_blas` (тянутся SYCL-бэкендом) требуют TBB 2023.1 — несовпадение символа `get_thread_reference_vertex`. Поэтому подключается только `/opt/intel/oneapi/setvars.sh`, никогда `/opt/intel/openvino/setvars.sh`.
- **`setvars.sh --force || true`.** Повторный source в той же сессии возвращает код 3 — это не ошибка сборки.

### 1.4. Отдельная ветка: OpenVINO/NPU

Параллельно собиралась версия с `GGML_OPENVINO=ON` (`build/ReleaseOV`. NPU Intel AI Boost виден и работает. Для LLM-инференса MoE-моделей уровня 30B NPU не даёт выигрыша против iGPU — оставлен как экспериментальный путь.

---

## 2. Тестирование бэкендов: успехи и провалы

### 2.1. Хронология

**Август (build 9722, GCC, Vulkan-only).** Бенчмарки Qwen3.8-27B (dense, Q4_K_M, 15.92 GiB) — файлы `~/tmp/bench_results*.txt`. Установлено: CPU-only даёт ~24 t/s prefill / 2.3 t/s decode; полный offload на iGPU (`-ngl 99`) — ~35 t/s prefill / ~2.5 t/s decode. KV-квантование (f16 → q8_0 → q4_0), flash attention, `--no-mmap` — в пределах погрешности, ±0.5 t/s. Вывод того периода: dense 27B на этом железе упирается в decode ~2.5 t/s, комфортного интерактива нет.

**04.09.2026.** Скачал Qwen3-Coder-30B-A3B (MoE, 3B активных) — ставка на то, что decode MoE считается по активным экспертам, а не по всем 30B. Пересборка под icpx + SYCL (build 10788).

**05.09.2026** Первый бенч новой модели: pp512 ≈ 130 t/s, tg128 ≈ 15.7 t/s. Но в логе — `ggml_sycl_init: no SYCL device available`: SYCL тихо упал, считал Vulkan. Загадка зафиксирована.

**05.09.2026** Расследование SYCL-падения — см. §2.3. Затем — серия сравнительных бенчей Vulkan vs SYCL на новой модели.

### 2.2. Итоговые цифры (Qwen3-Coder-30B-A3B Q4_K_M, ngl 99, fa, 5 повторов)

| Конфигурация                            | prefill (pp512), t/s | decode (tg128), t/s |
| --------------------------------------- | -------------------- | ------------------- |
| Vulkan, ub 256                          | 162.2 ± 2.3          | 15.67 ± 0.01        |
| Vulkan, ub 512                          | 214.5 ± 1.3          | 15.69 ± 0.02        |
| Vulkan, ub 1024                         | 212.8 ± 0.8          | 15.68 ± 0.01        |
| Vulkan, ub 512, t 12 (P-cores)          | 213.3 ± 1.4          | 15.72 ± 0.05        |
| SYCL, ub 256                            | 75.3 ± 0.9           | 17.84 ± 0.04        |
| SYCL, ub 512                            | 95.0 ± 0.9           | 18.05 ± 0.06        |
| SYCL, ub 512, tg512 (длинная генерация) | 95.0 ± 0.8           | 17.45 ± 0.09        |
| Vulkan, ub 512, tg512                   | 212.3 ± 2.8          | 15.46 ± 0.01        |

Контроль живым запуском (не бенч): длинный промпт — prefill 201 t/s, decode 14.3 t/s. Соответствует бенчу.

### 2.3. Расследование: почему SYCL молча падал

Симптом: `ggml_sycl_init: no SYCL device available` — при полностью рабочем стеке (`sycl-ls` видел `[level_zero:gpu] Intel Arc`, драйверы на месте).

Метод: бисекция окружения `env -i` + пофайловый `LD_LIBRARY_PATH` + `strace -e trace=openat` (скрипты `~/tmp/sycl_env_bisect*.sh`).

Причина: **SYCL-runtime делает runtime-`dlopen("libumf.so.1")`** (Unified Memory Framework, `/opt/intel/oneapi/umf/1.1/lib/`). Библиотека:

- отсутствует в зависимостях `ldd` (dlopen не виден статическому анализатору);
- не зарегистрирована в `ldconfig`;
- грузится только если umf-путь есть в `LD_LIBRARY_PATH` — что и делает `setvars.sh`.

Без неё SYCL-инициализация падает с вводящим в заблуждение сообщением «нет устройства», хотя устройство есть. Отсюда флаки: запуск из shell с setvars — работает; из чистого окружения (cron, systemd, ночной бенч) — тихо падает и откатывается на Vulkan.

Минимальный фикс без полного setvars:

```bash
LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib ./llama-bench ... --device SYCL0
```

### 2.4. Провал: гибрид «prefill на Vulkan + decode на SYCL» в одном процессе

Идея выглядела логично: Vulkan силён в prefill (2.2×), SYCL — в decode (+15%). Проверка исходников `ggml/src/ggml-backend.cpp` показала: планировщик llama.cpp назначает бэкенды **per-tensor** (по весам и буферам), фазовой логики prefill/decode не существует. Веса не могут одновременно лежать на двух устройствах; `--split-mode` делит по слоям, а не по фазам. Единственная альтернатива — два процесса с двумя копиями модели (2×18 ГБ RAM) и роутингом — признана бессмысленной: выигрыш decode всего +2 t/s.

### 2.5. Побочные наблюдения

- `-t 1` vs `-t 12`: при полном offload (`-ngl 99`) число CPU-потоков не влияет (0.02 t/s) — CPU делает только токенизацию и сэмплинг.
- ub 512 и 1024 эквивалентны по prefill; 512 выигрывает по VRAM под KV-кэш.
- `--no-mmap` на UMA ускоряет первый токен (модель в RAM, а не в page cache с догрузкой).
- KV-квантование (f16/q8_0/q4_0) на этой модели и iGPU — в пределах погрешности. KV по умолчанию оставлен f16.
- Дневной разброс pp512 у Vulkan (162–215) между запусками стабилизировался на ~213 после прогрева; ночные 130 t/s — вероятно, холодный page cache.

### 2.6. Тест двух новых моделей (06.09.2026)

**Qwen3.6-35B-A3B UD-Q4_K_M (20.60 GiB, 34.66B/3B активных), ngl 99, полный offload, r=3:**

| Бэкенд | pp512, t/s    | tg128, t/s   |
| ------ | ------------- | ------------ |
| Vulkan | 100.31 ± 0.36 | 11.41 ± 0.02 |
| SYCL   | 115.01 ± 0.44 | 13.64 ± 0.01 |

SYCL быстрее по обеим метрикам — впервые (у Coder-30B Vulkan выигрывал prefill 2.2×). Возможно, эффект UD-кванта или архитектуры.

**Qwen3-Coder-Next 80B.A3B Q4_K_M (45.19 GiB, 79.67B/3B активных):**

- Модель на деле **80B-A3B**, не 30B — в 64 GB RAM влезает только частичный offload.
- **Vulkan не работает вообще**: `ErrorOutOfDeviceMemory` на аллокации ~460–580 МБ при 42 ГБ свободных, при любом ngl (0–99) и любом batch. Причина: Mesa/Xe KMD пересчитывает бюджет после mmap 45-ГБ файла; SYCL/Level Zero аллоцирует иначе и работает.
- Потолок offload: **ngl 59 из ~60** (ngl 60 → системный OOM-killer «no killable processes»).

| ngl (SYCL)          | pp512, t/s       | tg128, t/s       |
| ------------------- | ---------------- | ---------------- |
| 0 (CPU)             | 13.18            | 1.29             |
| 20                  | 17.82            | 2.68             |
| 40                  | 30.90            | 7.31             |
| 48                  | 56.31            | 10.28            |
| 52                  | 70.44            | 11.55            |
| 58                  | 76.30 ± 0.76     | 11.90 ± 0.01     |
| **59 (финал, r=3)** | **76.48 ± 0.86** | **12.05 ± 0.02** |

Запуск SYCL требует `LD_LIBRARY_PATH=/opt/intel/oneapi/umf/1.1/lib` (см. §2.3).

---

## 3. Конечная конфигурация

### 3.1. Сборка

`~/ai/llama.cpp/build.sh` (см. §1.2): icpx + SYCL + Vulkan, GCC 15 toolchain, `-march=native -O3`. Один бинарник, оба GPU-бэкенда.

### 3.2. Интерактивный CLI — `~/ai/run_coder.sh`

```bash
exec llama-cli \
  -m .../Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  --device Vulkan0 \    # явно Vulkan: лучший prefill, не зависит от libumf
  -ngl 99 \             # все слои на iGPU
  -t 1 -tb 1 \          # CPU-потоки не важны при полном offload
  -c 32768 \            # контекст 32K
  -b 512 -ub 512 \      # ub 512 = prefill max, меньше VRAM чем 1024
  -fa auto \            # flash attention
  --no-mmap \           # UMA: модель в RAM
  --poll 100 --color auto
```

Без setvars.sh и oneAPI-переменных: Vulkan самодостаточен.

### 3.3. Сервер — `~/ai/run_coder_server.sh`

Та же схема плюс запуск сервера:

- `--jinja` — нативный chat template, **tool calling работает** (проверено: `finish_reason: tool_calls`, корректные JSON-аргументы);
- `--alias qwen3-coder-30b` — стабильный model id для клиентов;
- `--cache-reuse 256` — повторное использование KV префикса (агентные клиенты пересылают system prompt каждый запрос);
- `--host 127.0.0.1 --port 8080`.

### 3.4. systemd — `~/.config/systemd/user/llama-coder.service`

```ini
[Service]
ExecStart=<user>/ai/run_coder_server.sh
Restart=on-failure
RestartSec=10
TimeoutStartSec=300      # загрузка 18GB модели ~60с, запас
Nice=-5
MemoryHigh=52G
MemoryMax=58G             # защита от OOM при 60G RAM
```

`WantedBy=default.target`, enabled, `loginctl Linger=yes` — стартует при загрузке без логина. Проверено: `systemctl --user` → active, health OK, kilo отвечает.

### 3.5. Клиент — kilo CLI (`~/.config/kilo/kilo.jsonc`)

```jsonc
{
  "model": "openai-compatible/qwen3-coder-30b",
  "provider": {
    "openai-compatible": {
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen3-coder-30b": {
          "name": "Qwen3-Coder 30B-A3B (local Vulkan)",
          "tool_call": true,
          "temperature": true,
          "limit": { "context": 32768, "output": 16384 }
        }
      }
    }
  }
}
```

`limit.context` обязателен: без него kilo отключает автокомпакцию диалога (документация kilo). Проверено end-to-end: kilo → сервер → модель → tool calls (создание/чтение файлов агентом).

---

## 4. Выводы

1. **Для Arc MTL iGPU Vulkan — лучший рабочий бэкенд**: prefill 213 t/s (2.2× быстрее SYCL), decode 15.7 t/s. SYCL выигрывает только в decode (+15%, 18 t/s) — не окупается потерей prefill.
2. **MoE — правильный класс моделей для iGPU**: Qwen3-Coder-30B-A3B (3B активных) даёт 15.7 t/s decode против 2.5 t/s у dense Qwen3.8-27B — 6× разница при большем общем размере.
3. **Гибрид prefill/decode по бэкендам в llama.cpp невозможен**: планировщик per-tensor, фазовой логики нет. Проверено по исходникам, не по документации.
4. **SYCL-стек Intel хрупок в окружении**: runtime-dlopen `libumf.so.1` вне ldconfig делает инициализацию зависимой от `LD_LIBRARY_PATH` от setvars.sh. Симптом («нет устройства») лжёт о причине. Для демонов/cron — либо Vulkan, либо явный umf-путь в LD_LIBRARY_PATH.
5. **CPU-потоки и KV-квантование не влияют** при полном offload на iGPU: `-t 1`, KV f16 — достаточно.
6. **Итоговая цепочка**: build.sh (icpx+SYCL+Vulkan) → run_coder_server.sh (Vulkan, jinja, cache-reuse) → systemd user unit (autostart, OOM-guard) → kilo (openai-compatible, tool_call). Полностью локальный кодинг-стек: 32K контекст, ~213 t/s prefill, ~15.7 t/s decode, tool calling.