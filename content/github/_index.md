+++
title = 'Мой GitHub'
description = 'Мои программы, в рамках хобби'
date = 2026-09-09
draft = false
tags = ["GitHub"]
categories =  ["GitHub"]
[menus] 
    [menus.main]
        name="Мой GitHub"
        weight = 5
+++

## [light_web](https://github.com/smobidin/light_web)

Легковесный одностраничный веб-сервер на Python и на Go, для просмотра Markdown-документов, исходного кода и директорий в браузере. Поддерживает подсветку синтаксиса 500+ языков через Pygments и рендеринг LaTeX-формул (KaTeX). Go версия компилируется в один файл. Лицензия Apache-2.0. Подходит для быстрой навигации по документации или кодовой базе в локальной сети.

## [speech_tools](https://github.com/smobidin/speech_tools)

Оффлайн-тулкит для транскрибации речи (ASR) и диаризации спикеров с веб-интерфейсом. Использует faster-whisper (int8 CPU) для распознавания речи и SpeechBrain ECAPA-модели для извлечения эмбеддингов спикеров. Включает Flask-based WebUI с управлением записями, настройками, отслеживанием прогресса и панелью отладки. Поддерживает 6 размеров моделей от tiny до large-v3-turbo, полностью работает локально без интернета.

## [dotfiles](https://github.com/smobidin/dotfiles)

Личные конфигурационные файлы (dotfiles), управляемые через bare git-репозиторий. Стек: i3wm, zsh (плагины через antidote), vim (плагины через pathogen), polybar, kitty, rofi. Включает 277 коммитов, набор CLI-утилит (bat, fd, ripgrep, fzf, delta) и настройки темы (pywal, vivid). Основан на Debian-based дистрибутивах Linux.