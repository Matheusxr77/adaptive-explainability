# Explicabilidade em Aprendizado de Máquina - Q1.1

Projeto da disciplina Explicabilidade em Aprendizado de Máquina (UFAPE 2026).

Objetivo da Questão 1.1: descobrir, para cada instância de teste, o menor número de perturbações do LIME que mantém a explicação estável e coerente, usando SLM local via Docker Model Runner.

## Requisitos

- Windows 10/11
- Python 3.11+
- Docker Desktop atualizado e em execução
- Dataset em `data/credit_risk_dataset.csv`

## Execução Obrigatória com Docker Model Runner

Neste projeto, o uso de SLM local via Docker Model Runner é requisito obrigatório.
Ou seja: para execução oficial, o modelo `ai/qwen2.5` deve estar ativo durante o pipeline.

Endpoint esperado pelo código: `http://localhost:12434/engines/llama.cpp/v1`

## Passo a passo completo no VS Code

### 1. Abrir a pasta do projeto

No terminal integrado do VS Code:

```bash
cd adaptative-explainability
```

### 2. Criar e preparar o ambiente Python

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/pip.exe install -r requirements.txt
```

### 3. Baixar o modelo local (uma vez)

```bash
docker model pull ai/qwen2.5
```

### 4. Iniciar o SLM local

Em um terminal dedicado, execute:

```bash
docker model run ai/qwen2.5
```

Comportamento esperado: o terminal entra em modo interativo (`Send a message`) e permanece ativo.
Mantenha este terminal aberto durante toda a execução do pipeline.

### 5. Rodar o pipeline em outro terminal

```bash
.venv/Scripts/python.exe main.py
```

Ordem de execução:
1. `scripts/00_eda.py`
2. `scripts/01_modelo_baseline.py`
3. `scripts/02_lime_base.py`
4. `scripts/03_questao_1_1.py`
5. `scripts/04_relatorio_final.py`

Tempo médio: 15 a 40 minutos, variando conforme hardware e carga do SLM.

### 6. Validar resultados

```bash
.venv/Scripts/python.exe tests.py
```

## Como confirmar que o Docker foi realmente usado

Após o `main.py`, verifique:

- `resultados/cache/llm_prompts.csv`
- `resultados/cache/llm_responses.json`
- `resultados/csv/q11_text_explanations.csv`

Se as respostas textuais estiverem preenchidas e sem status predominante `offline_skip`, o SLM foi utilizado corretamente.

## Estrutura do projeto

```text
explicabilidade-am/
	data/
		credit_risk_dataset.csv
	models/
	resultados/
		csv/
		figuras/
		cache/
	relatorio/
	scripts/
		00_eda.py
		01_modelo_baseline.py
		02_lime_base.py
		03_questao_1_1.py
		04_relatorio_final.py
		data_loader.py
	src/
		config.py
		io_utils.py
		plotting.py
		llm_client.py
		explainer.py
		evaluation.py
	main.py
	tests.py
	requirements.txt
```

## Solução de problemas

### Erro de espaço em disco ao baixar modelo

Se aparecer erro de falta de espaço, libere ao menos 8 a 10 GB no `C:` e repita o pull.

### Erro `resumable: exceeded retry budget`

Use este procedimento de limpeza:

```bash
docker model rm ai/qwen2.5 || true
rm -rf "$HOME/.docker/models"
```

Depois, reinicie o Docker Desktop e rode novamente:

```bash
docker model pull ai/qwen2.5
```

### `Activate.ps1` falhando no terminal bash

No bash, não use `Activate.ps1`. Execute diretamente com o Python do ambiente:

```bash
.venv/Scripts/python.exe main.py
```

## Resultados esperados

Ao final da execução, os arquivos principais são:

- `models/rf_model.joblib`
- `models/preprocessor.joblib`
- `resultados/csv/q11_raw_runs.csv`
- `resultados/csv/q11_adaptive_n_by_instance.csv`
- `resultados/csv/q11_n_level_summary.csv`
- `resultados/figuras/*.png`
- `relatorio/relatorio_final.md`

## Reprodutibilidade

O projeto usa `RANDOM_STATE = 42` e sementes derivadas por repetição (`42 + rep_idx`).
