+++
title = 'Запускаем llama.cpp на RISC-V VisionFive 2'
date = 2026-03-01T17:59:55+03:00
draft = false
tags = ["AI", "RISC-V", "HW"]
categories =  ["Искусcтвенный Интеллект", "Инфраструктура"]
+++

## Пробуем запускать LLM на RISC-V

![banner](/img/llama_riscv_ai_banner.png)

Целью эксперимента было не столько проверить производительность, сколько понять применимость процессоров RISC-V в качестве управляющих в серверах для ИИ.

Компания Nvidia использует ARM процессоры [Vera](https://www.nvidia.com/en-us/data-center/vera-cpu/) в качестве управляющих для GPU [Rubin](https://www.nvidia.com/en-us/data-center/technologies/rubin/).

Почему бы не попробовать использовать RISC-V?

В качестве инференес-движка выбрал [LLaMA C++ - LLM inference in C/C++](https://github.com/ggml-org/llama.cpp)

Критерием успеха для себя выбрал: **модель LLM работает и ответила мне хотя бы одним словом**.

## На чем пробовал и как собирал

Одноплатник StarFive VisionFive 2:

``` console
$ fastfetch
                  -`                     stas@vf2a
                 .o+`                    ---------
                `ooo/                    OS: Arch Linux riscv64
               `+oooo:                   Host: StarFive VisionFive 2 CM
              `+oooooo:                  Kernel: Linux 6.12.18-cwt-6.0.0-2
              -+oooooo+:                 Uptime: 21 mins
            `/:-:++oooo+:                Packages: 554 (pacman)
           `/++++/+++++++:               Shell: zsh 5.9
          `/++++++++++++++:              Terminal: /dev/pts/0
         `/+++ooooooooooooo/`            CPU: jh7110s (4) @ 1.50 GHz
        ./ooosssso++osssssso+`           GPU: img-gpu [Integrated]
       .oossssso-````/ossssss+`          Memory: 370.64 MiB / 7.71 GiB (5%)
      -osssssso.      :ssssssso.         Swap: 0 B / 3.85 GiB (0%)
     :osssssss/        osssso+++.        Disk (/): 13.47 GiB / 58.13 GiB (23%) - btrfs
    /ossssssss/        +ssssooo/-        Local IP (wlp1s0u1u3): 10.1.1.110/24
  `/ossssso+/:-        -:/+osssso+-      Locale: en_US.UTF-8
 `+sso+:-`                 `.-/+oso:
`++:.                           `-/+/
.`                                 `/
```

## Неудачи

Пробовал собирать из ветки master, на момент `git clone`. Собирал с поддержкой CPU и OpenBLAS.

```bash
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DBUILD_SHARED_LIBS=OFF
```

Компиляция не прошла. Билдер некорректно определил набор инструкций процессора. Процессор на плате **StarFive VisionFive 2** -  **jh7110s**. Его набор инструкций:

``` bash
$ cat /proc/cpuinfo | tail | grep 'hart isa'
hart isa    : rv64imafdc_zicntr_zicsr_zifencei_zihpm_zca_zcd_zba_zbb
```

Билдер же прописывал следующий набор инструкций:

``` bash
Adding CPU backend variant ggml-cpu: -march=rv64gc_zfh_v_zvfh_zicbop_zihintpause;-mabi=lp64d
```

Пробовал отключить поддержку RVV, однако компиляция так и шла с ошибками:

``` bash
$ cmake -B build -DGGML_BLAS=ON  \
-DGGML_BLAS_VENDOR=OpenBLAS  \
-DBUILD_SHARED_LIBS=OFF  \
-DGGML_RVV=OFF  \
-DGGML_RV_ZFH=OFF  \
-DGGML_RV_ZVFH=OFF  \
-DGGML_RV_ZICBOP=OFF  \
-DGGML_RV_ZIHINTPAUSE=OFF
```

Компиляция падала в ошибку:

```bash
ggml-cpu/arch/riscv/quants.c:1979:5: error: unknown type name ‘vuint16mf2_t’; did you mean ‘uint16_t’?
 1979 |     vuint16mf2_t v_gather_qh = __riscv_vle16_v_u16mf2(gather_qh_arr, 8);
      |     ^~~~~~~~~~~~
      |     uint16_t
```

Провел много экспериментов с билд-ключами, все безуспешные. После этого всмотрелся в код `ggml-cpu/arch/riscv/quants.c` и обнаружил, что апстрим ветка жестко требует наличия RVV 1.0 и все переменные в коде имеют префикс `v`. Т.е. разработчики llama.cpp явно жестко прописали в коде поддержку только новых процессоров RISC-V с полной реализацией спецификации [RVV 1.0](https://lists.riscv.org/g/tech-vector-ext/attachment/691/0/riscv-v-spec-1.0.pdf).

В репозитории [GitHub ggml-org / llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master) нашел файл [build-riscv64-spacemit.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build-riscv64-spacemit.md) и по нему понял, почему в коде `ggml-cpu/arch/riscv/quants.c` так жестко прописали `RVV 1.0`

Ну что же, похоже, что следующей моей платой для экспериментов станет плата с процессором **SpacemiT K3**:

``` bash
model name      : Spacemit(R) X60
isa             : rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt
mmu             : sv39
uarch           : spacemit,x60
mvendorid       : 0x710
marchid         : 0x8000000058000001
```

## Успех

А пока решил откатиться на более ранний коммит и собрать уже не с OpenBLAS, а с **Vulkan**.

Откатил лламу на ветку <https://github.com/ggml-org/llama.cpp/releases/tag/b6900>

Пробую собрать с ключами <https://github.com/ggml-org/llama.cpp/issues/14926>

``` bash
cmake -S . -B build  \
-DGGML_RVV=OFF   \
-DBUILD_SHARED_LIBS=OFF  \
-DGGML_Vulkan=ON   \
-DGGML_NO_LLAMAFILE=ON  \
-DGGML_RV_ZFH=OFF  \
-DGGML_RV_ZICBOP=OFF  \
-DCMAKE_BUILD_TYPE=Release  \
-DCMAKE_C_FLAGS="-march=rv64imafdc"  \
-DCMAKE_CXX_FLAGS="-march=rv64imafdc"
```

Собираю с ключами:

``` bash
cmake --build build --config Release -j 4
```

Сборка пошла без ошибок:

![сборка llama.cpp](/img/llama_riscv-1.png)

Собралось без ошибок. Пробую скачать и запустить модель:

``` bash
./llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --host 0.0.0.0 --port 8080
```

Скачивание идет успешно:

![скачивание Qwen3](/img/llama_riscv-2.png)

И начался запуск модели:

![запуск Qwen3](/img/llama_riscv-3.png)

Модель запустилась, первый промпт приняла, начала генерировать ответ. Скажу сразу, поскольку все село на Vulkan и встроенную GPU, то скорость генерации токенов не то, что низкая, она почти никакая (3-5 токена в **минуту**).

## Вердикт

Оно работает ;).

Просто убедился в этом на собственном эксперименте.

Да, предстоит еще большой путь в развитии RISC-V, но даже те процессоры и системы на их базе, которые уже есть на рынке - на мой взгляд уже применимы в качестве платформ для ИИ.

Как в edge inference, так и в построении кластеров с GPU.

## Что из плат посмотреть

| Производитель | Название платы                                           | Модель процессора      | Краткие характеристики (ядра, такт, память, набор инструкций и технологии)                                                                                                                                                                                                                                                                                                                                                                                                     | Ссылка                            |
|---------------|----------------------------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| Sipeed        | Lichee K3 Pico‑ITX (eval‑плата для K3, семейство Lichee) | SpacemiT K3            | 8× RISC‑V ядра (линейка X100/A100, до 2.4–2.5 ГГц по материалам по K3), 64‑бит, профиль RVA23, поддержка RVV 1.0 с векторами до 1024 бит, аппаратная виртуализация (RV Hypervisor 1.0, AIA, IOMMU по K3), AI‑ускоритель до ~60 TOPS int8, GPU IMG BXM‑4‑64‑MC1, LPDDR5 до 32 ГБ, UFS + NVMe (PCIe Gen3), MMU sv39; ISA: базовый RV64IMAFDCH + B, V (RVV 1.0), расширения для гипервизора и безопасной виртуализации, соответствует профилю RVA23.                              | <https://sipeed.com/k3>           |
| Milk‑V        | Jupiter (Mini‑ITX)                                       | SpacemiT K1/M1         | 8× X60 ядра, RV64GCVB, профиль RVA22, RVV 1.0, до 1.6–1.8 ГГц в зависимости от K1/M1, 2 TOPS NPU, GPU IMG BXE‑2‑32, LPDDR4X 4–16 ГБ, PCIe 2.0 x2 (слот x8), dual GbE, Wi‑Fi 6/BT 5.2, HDMI до 1920×1440; ISA: RV64IMAFDCH + B, V (RVV 1.0) в рамках профиля RVA22.                                                                                                                                                                                                             | <https://milkv.io/ru/jupiter2>    |
| Milk‑V        | Jupiter2 (SBC)                                           | SpacemiT K3            | 8× X100™ CPU @ до 2.4 ГГц, 64‑бит RISC‑V AI‑ядра, профиль RVA23, RVV 1.0 до 1024‑бит, отдельный 8‑ядерный A100™ AI‑движок до 60 TOPS (INT4/8, FP8/16, BF16), GPU IMG BXM‑4‑64‑MC1, LPDDR5 до 32 ГБ, UFS до 256 ГБ + M.2 NVMe (PCIe Gen3 x4), 10GbE SFP+ + GbE + Wi‑Fi 6/BT 5.2, USB‑C DP Alt + eDP; ISA: RV64IMAFDCH + B, V (RVV 1.0, 1024‑бит), аппаратная виртуализация (RV Hypervisor 1.0, AIA, IOMMU), профиль RVA23.                                                      | <https://milkv.io/ru/jupiter2>    |
| Milk‑V        | Jupiter NX (SoM)                                         | SpacemiT K1/M1         | 8‑ядерный SoM на SpacemiT K1/M1, профиль RVA22, поддержка RVV 1.0, LPDDR4X, eMMC на модуле, совместим с базовыми платами Nano/Xavier NX; ISA: RV64IMAFDCH + B, V (RVV 1.0) по RVA22.                                                                                                                                                                                                                                                                                           | <https://milkv.io/jupiter-nx>     |
| Milk‑V        | Jupiter2 NX (SoM)                                        | SpacemiT K3            | 8‑ядерный X100™ CPU @ 2.4 ГГц, 64‑бит, профиль RVA23, RVV 1.0 до 1024‑бит, отдельный 8‑ядерный A100™ AI‑ускоритель до 60 TOPS, GPU IMG BXM‑4‑64‑MC1, LPDDR5 до 32 ГБ, onboard UFS (до 256 ГБ), поддержка NVMe (PCIe Gen3), DP 1.2 + MIPI DSI, 2× MIPI CSI, GbE, USB 3.0/2.0, −40…+85 °C, TDP 18–35 Вт; ISA: RV64IMAFDCH + B, V (RVV 1.0, 1024‑бит), аппаратная виртуализация (RV Hypervisor 1.0, AIA, IOMMU), профиль RVA23.​                                                   | <https://milkv.io/ru/jupiter2-nx> |
| Milk‑V        | Pioneer (mATX плата)                                     | SOPHON (SOPHGO) SG2042 | 64‑ядерный RISC‑V C920 до 2 ГГц, RVV 0.71, 4‑канальный DDR4 (4 DIMM, до 128 ГБ, 3200 MT/s, ECC RDIMM/UDIMM/SODIMM), 2× PCIe 4.0 x16 (через CCIX и PCIe‑коммутатор), доп. слот x8, 5× SATA3, 8× USB3, 2× M.2 M‑Key (PCIe 3.0 x4), 1× M.2 E‑Key (PCIe 3.0 x1 + USB 2.0), 2× 2.5G RJ45, microSD, eMMC модуль, ATX 24‑pin; ISA: 64‑битный RISC‑V с поддержкой RVV 0.71 (векторное расширение до финализации RVV 1.0), стандартные расширения IMACF/D (C920‑класс), MMU для Linux.​  | <https://milkv.io/ru/pioneer>     |
