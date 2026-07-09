.PHONY: all setup-envs test-all mypy-all quick-test quick-mypy stress stress-quick check-versions clean help

# Версии Python для тестирования через uv
PYTHON_VERSIONS = 3.9 3.10 3.11 3.12 3.13 3.14

# Цель по умолчанию
all: test-all mypy-all

# Настройка окружений для всех версий Python
setup-envs:
	@echo "=== Настройка окружений для всех версий Python ==="
	@for version in $(PYTHON_VERSIONS); do \
		echo "--- Настройка окружения Python $$version ---"; \
		VIRTUAL_ENV=.venv-$$version uv sync --active --python $$version --extra dev || echo "⚠️  Ошибка настройки для Python $$version"; \
		echo ""; \
	done

# Тестирование на всех версиях Python
test-all: setup-envs
	@echo "=== Запуск тестов на всех версиях Python ==="
	@for version in $(PYTHON_VERSIONS); do \
		echo "--- Тестирование Python $$version ---"; \
		VIRTUAL_ENV=.venv-$$version uv run --active --python $$version pytest -q tests/ || echo "❌ Тесты Python $$version завершились с ошибкой"; \
		echo ""; \
	done

# Проверка mypy на всех версиях Python
mypy-all: setup-envs
	@echo "=== Запуск mypy на всех версиях Python ==="
	@for version in $(PYTHON_VERSIONS); do \
		echo "--- Mypy проверка Python $$version ---"; \
		VIRTUAL_ENV=.venv-$$version uv run --active --python $$version mypy --cache-dir .mypy_cache-$$version aiologging/ || echo "❌ Mypy Python $$version завершился с ошибкой"; \
		echo ""; \
	done

# Динамические цели для каждой версии Python
define test-python-template
.PHONY: test-python$(1)
test-python$(1):
	@echo "--- Тестирование Python $(1) ---"
	VIRTUAL_ENV=.venv-$(1) uv run --active --python $(1) pytest tests/ -v
endef

define mypy-python-template
.PHONY: mypy-python$(1)
mypy-python$(1):
	@echo "--- Mypy проверка Python $(1) ---"
	VIRTUAL_ENV=.venv-$(1) uv run --active --python $(1) mypy --cache-dir .mypy_cache-$(1) aiologging/
endef

$(foreach version,$(PYTHON_VERSIONS),$(eval $(call test-python-template,$(version))))
$(foreach version,$(PYTHON_VERSIONS),$(eval $(call mypy-python-template,$(version))))

# Быстрые проверки на одной версии (последней из списка)
QUICK_VERSION = $(lastword $(PYTHON_VERSIONS))

quick-test:
	VIRTUAL_ENV=.venv-$(QUICK_VERSION) uv run --active --python $(QUICK_VERSION) pytest -q tests/

quick-mypy:
	VIRTUAL_ENV=.venv-$(QUICK_VERSION) uv run --active --python $(QUICK_VERSION) mypy --cache-dir .mypy_cache-$(QUICK_VERSION) aiologging/

# Стресс-тесты (см. docs/stress-testing.md)
stress:
	VIRTUAL_ENV=.venv-$(QUICK_VERSION) uv run --active --python $(QUICK_VERSION) python -m stress run --json logs/stress-report.json

stress-quick:
	VIRTUAL_ENV=.venv-$(QUICK_VERSION) uv run --active --python $(QUICK_VERSION) python -m stress run --quick

# Проверка установленных версий Python через uv
check-versions:
	@echo "=== Проверка установленных версий Python ==="
	@for version in $(PYTHON_VERSIONS); do \
		if uv python find $$version >/dev/null 2>&1; then \
			echo "✅ Python $$version установлен через uv"; \
		else \
			echo "❌ Python $$version не установлен через uv"; \
		fi; \
	done

# Очистка кэшей и окружений
clean:
	@echo "Очистка кэшей и окружений..."
	rm -rf .pytest_cache/ .pytest_cache-*/
	rm -rf .mypy_cache/ .mypy_cache-*/
	rm -rf .coverage .coverage-*
	rm -rf htmlcov/ htmlcov-*/
	rm -rf aiologging.egg-info/
	rm -rf .venv-*/

# Помощь
help:
	@echo "Доступные команды:"
	@echo "  make all              - Запустить все тесты и mypy проверки"
	@echo "  make test-all         - Запустить pytest на всех версиях Python"
	@echo "  make mypy-all         - Запустить mypy на всех версиях Python"
	@echo "  make test-python3.X   - Запустить pytest на конкретной версии Python"
	@echo "  make mypy-python3.X   - Запустить mypy на конкретной версии Python"
	@echo "  make quick-test       - Быстрый тест на Python $(QUICK_VERSION)"
	@echo "  make quick-mypy       - Быстрая проверка mypy на Python $(QUICK_VERSION)"
	@echo "  make stress           - Полный стресс-прогон (минуты, JSON в logs/)"
	@echo "  make stress-quick     - Быстрый стресс-прогон (~10 секунд)"
	@echo "  make setup-envs       - Настроить окружения для всех версий Python"
	@echo "  make check-versions   - Проверить установленные версии Python"
	@echo "  make clean            - Очистить кэши и временные файлы"
	@echo "  make help             - Показать эту справку"
