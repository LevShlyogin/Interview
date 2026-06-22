# Agentic Knowledge Layer

Корпоративный ИИ-ассистент (RAG-система) для компании «Брусника». Система предназначена для поиска и агрегации информации по внутренней базе знаний (регламенты, процессы, стандарты) с учетом уровней доступа пользователей и мультидоменной архитектуры.

## Ключевые архитектурные решения (Highlights)

* **Hybrid LLM Cascading:** Разделение когнитивной нагрузки. Легкая локальная модель (Qwen 2.5 3B) выполняет быстрый роутинг и оценку контекста, а тяжелая модель генерирует финальный ответ.
* **Agentic RAG (LangGraph):** Процесс поиска обернут в конечный автомат с узлами рерайтинга запросов, маршрутизации, оценки контекста и Fallback-логикой для уточнения неоднозначных вопросов.
* **Hybrid Search + Reranking:** Векторная база Qdrant использует одновременно семантический поиск (Dense векторы) и точный текстовый поиск (BM25). Результаты объединяются через Reciprocal Rank Fusion (RRF) и переранжируются через Cross-Encoder (`msmarco`).
* **Deterministic Validator:** Нативный Post-Processing фильтр на регулярных выражениях (Regex), который вычищает галлюцинации LLM, удаляет markdown-ссылки и предотвращает утечку внутренних имен файлов.
* **Smart Chunking & Table Extraction:** Нарезка Markdown с сохранением иерархии заголовков. Таблицы извлекаются, заменяются на плейсхолдеры, а локальная LLM генерирует для них текстовые саммари перед векторизацией.

## Технологический стек

* **Оркестрация:** LangChain, LangGraph
* **Векторная БД:** Qdrant (Docker)
* **Эмбеддинги:** FastEmbed (`paraphrase-multilingual-MiniLM-L12-v2` + `Qdrant/bm25`)
* **Reranker:** `DiTy/cross-encoder-russian-msmarco`
* **LLMs:** Ollama (Qwen 2.5 3B), OpenRouter API
* **Интерфейс:** Streamlit
* **Пакетный менеджер:** Poetry

## Структура проекта

```text
Interview/
├── 📁 data/
│   └── 📁 knowledge_base/       # Markdown-файлы базы знаний
├── 📁 src/
│   └── 📁 brusnika_knowledge_layer/
│       ├── 📁 database/         # Управление подключениями (Qdrant)
│       ├── 📁 graph/            # Логика LangGraph (узлы, стейт, воркфлоу)
│       ├── 📁 ingestion/        # Парсинг, чанкинг, извлечение таблиц
│       ├── 📁 retrieval/        # Гибридный поиск и реранжирование
│       └── 🐍 app.py            # Streamlit интерфейс
├── 📝 ARCHITECTURE.md           # Архитектурная записка
├── 🐳 docker-compose.yml        # Конфигурация Qdrant
├── 🔒 poetry.lock               
├── 📦 pyproject.toml            
└── 📖 README.md                 

```

## Быстрый старт (Установка и запуск)

### 1. Подготовка окружения

Убедитесь, что у вас установлены: `Python >= 3.10`, `Poetry`, `Docker` и `Ollama`.

Скачайте локальную модель для роутинга и суммаризации:

```bash
ollama pull qwen2.5:3b

```

Установите зависимости проекта:

```bash
poetry install

```

### 2. Запуск инфраструктуры

Поднимите локальный инстанс Qdrant:

```bash
docker-compose up -d

```

Настройте переменные окружения. Создайте файл `.env` в корне проекта:

```env
OPENROUTER_API_KEY=sk-or-v1-ваш-ключ

```

### 3. Ингестия (Загрузка базы знаний)

Для обработки Markdown-файлов и загрузки их в Qdrant запустите скрипт:

```bash
poetry run python -m brusnika_knowledge_layer.ingestion.pipeline

```

### 4. Запуск интерфейса

Запустите Streamlit-приложение для взаимодействия с ассистентом:

```bash
poetry run streamlit run src/brusnika_knowledge_layer/app.py