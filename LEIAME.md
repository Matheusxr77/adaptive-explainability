# 📖 LEIAME - Explicabilidade Adaptativa

## 🎯 Sobre o Projeto

Este projeto implementa técnicas de explicabilidade adaptativa para modelos de Machine Learning, combinando métodos tradicionais (LIME/SHAP) com Small Language Models (SLMs) locais. O objetivo é responder duas questões principais de pesquisa:

**Task 1.1**: Como escolher adaptativamente o número ótimo de perturbações do LIME para obter explicações coerentes?

**Task 1.2**: É possível agregar múltiplas explicações "fracas" (poucas perturbações) em uma única explicação "forte" com menor custo computacional?

## 🚀 Instalação Rápida

### Pré-requisitos

- **Python 3.9+** (testado com Python 3.13.3)
- **Docker Desktop** (para executar os SLMs localmente)
- **8GB+ RAM** recomendado para os modelos de linguagem
- **Windows, Linux ou Mac**

### Passo 1: Clonar/Baixar o Projeto

Se você já tem o projeto, pule para o Passo 2.

### Passo 2: Instalar Dependências Python

Abra um terminal na pasta do projeto e execute:

```bash
pip install -r requirements.txt
```

Isso instalará:
- `xgboost` - Modelo de classificação
- `lime` - Explicador baseado em perturbações
- `shap` - Explicador baseado em valores de Shapley
- `pandas`, `numpy` - Manipulação de dados
- `matplotlib`, `seaborn`, `plotly` - Visualizações
- `scikit-learn` - Utilitários de ML
- `requests` - Comunicação com SLMs

### Passo 3: Configurar Docker e SLMs

Os Small Language Models serão executados localmente via Docker. Escolha o script apropriado para seu sistema:

#### Windows (PowerShell)

```powershell
cd configs
.\docker_setup.ps1
```

#### Linux/Mac (Bash)

```bash
cd configs
bash docker_setup.sh
```

**O que esse script faz:**
1. Verifica se o Docker está instalado e rodando
2. Baixa os modelos `ai/qwen2.5` e `ai/granite-4.0-h-nano`
3. Inicia containers nas portas 8080 (Qwen - primário) e 8081 (Granite - backup)
4. Testa a conectividade dos endpoints

**Tempo estimado:** 5-10 minutos (dependendo da velocidade da internet)

### Passo 4: Verificar Instalação

Execute o script de teste:

```bash
python test_setup.py
```

Você deve ver:
- ✓ Python 3.9+ detectado
- ✓ Todas as bibliotecas instaladas
- ✓ Dataset encontrado (credit_risk_dataset.csv)
- ✓ Docker containers rodando
- ✓ SLMs respondendo

Se houver erros, consulte a seção de Solução de Problemas abaixo.

## 🎮 Como Executar

O projeto oferece três modos de execução:

### Modo 1: Demo Rápido (5-10 minutos)

Ideal para testar o sistema rapidamente.

```bash
python main.py --modo demo
```

**O que faz:**
- Usa apenas 5 instâncias de teste
- Executa Task 1.1 com 3 amostras
- Executa Task 1.2 com 3 amostras
- Gera relatório resumido

### Modo 2: Análise Completa (1-2 horas)

Análise completa com estatísticas robustas.

```bash
python main.py --modo completo
```

**O que faz:**
- Usa 200+ instâncias de teste
- Executa Task 1.1 com 30 amostras
- Executa Task 1.2 com 30 amostras
- Gera relatório detalhado com todos os gráficos

### Modo 3: Sem SLM (rápido, apenas explainers)

Executa apenas o modelo e explainers, sem usar SLMs.

```bash
python main.py --modo sem-slm
```

**Útil para:**
- Testar quando SLMs não estão disponíveis
- Comparar LIME vs SHAP sem overhead do LLM
- Debugging rápido

### Opções Adicionais

```bash
# Usar dataset diferente
python main.py --modo completo --dados "data/meu_dataset.csv"

# Salvar saídas em outro diretório
python main.py --modo demo --saida "resultados/teste1"

# Usar SLM em outro servidor
python main.py --modo demo --slm-url "http://meu-servidor:9000"
```

## 📊 Interpretando os Resultados

Todos os arquivos são salvos em `outputs/` (ou diretório especificado):

### Gráficos Principais

1. **data_exploration.png**
   - Distribuições das features
   - Correlações entre variáveis
   - Balanceamento das classes

2. **model_performance.png**
   - Curva ROC
   - Curva Precision-Recall
   - Matriz de confusão
   - Métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC

3. **feature_importance.png**
   - Ranking de importância das features pelo XGBoost
   - Identifica quais variáveis mais influenciam as predições

4. **task1.1_adaptive_selection.png**
   - Comparação entre busca sequencial e binária
   - Perturbações necessárias vs threshold de coerência
   - Redução no número de perturbações

5. **task1.2_aggregation_results.png**
   - Comparação entre estratégias de agregação (concatenação vs síntese)
   - Métricas de qualidade: Jaccard similarity, concordância direcional
   - Análise de custo: N×fracas vs 1×forte

6. **lime_vs_shap_comparison.png**
   - Comparação direta entre explicações LIME e SHAP
   - Alinhamento de features importantes
   - Diferenças nos valores de importância

### Arquivos CSV

- `adaptive_selection_results.csv` - Dados detalhados da Task 1.1
- `aggregation_results.csv` - Dados detalhados da Task 1.2
- `model_metrics.csv` - Métricas do modelo XGBoost

### Relatório Textual

- `summary_report.txt` - Resumo executivo com:
  - Performance do modelo
  - Resultados das Tasks 1.1 e 1.2
  - Economias de custo computacional
  - Conclusões e recomendações

## 🐛 Solução de Problemas Comuns

### Erro: "ModuleNotFoundError: No module named 'xgboost'"

**Causa:** Dependências não instaladas.

**Solução:**
```bash
pip install -r requirements.txt
```

### Erro: "Connection refused" ao contactar SLM

**Causa:** Docker containers não estão rodando.

**Solução:**
```bash
# Verificar status
docker ps

# Se não aparecer qwen2.5-server e granite-nano-server:
cd configs
.\docker_setup.ps1  # Windows
bash docker_setup.sh  # Linux/Mac
```

### Erro: "Docker daemon is not running"

**Causa:** Docker Desktop não está aberto.

**Solução:** Abra o Docker Desktop e aguarde inicialização completa (ícone deve ficar verde).

### Erro: "Out of memory" no Docker

**Causa:** Limite de memória do Docker muito baixo.

**Solução:**
1. Abra Docker Desktop
2. Settings → Resources → Memory
3. Aumente para pelo menos 8GB
4. Clique em "Apply & Restart"

### Performance Muito Lenta

**Possíveis causas e soluções:**

1. **Muitas amostras:** Use `--modo demo` primeiro
2. **Busca sequencial lenta:** Edite `main.py` para usar `strategy='binary'`
3. **Muitos testes de estabilidade:** Reduza `stability_runs` de 3 para 2
4. **SLM sobrecarregado:** Reinicie o container:
   ```bash
   docker restart qwen2.5-server
   ```

### Dataset Não Encontrado

**Erro:** `FileNotFoundError: data/credit_risk_dataset.csv`

**Solução:** Certifique-se que o arquivo CSV está na pasta `data/`. Ou especifique o caminho:
```bash
python main.py --modo demo --dados "caminho/para/seu/dataset.csv"
```

### Gráficos Não Aparecem

**Causa:** Backend do matplotlib não configurado.

**Solução:** Os gráficos são salvos como PNG em `outputs/`. Abra os arquivos diretamente.

## 🔧 Configurações Avançadas

### Ajustar Thresholds de Coerência

Edite `main.py`, procure por:

```python
selector = AdaptivePerturbationSelector(
    coherence_threshold=7.0,  # ← Ajuste aqui (escala 0-10)
    variance_threshold=0.15,  # ← Ajuste aqui
    # ...
)
```

- **coherence_threshold**: Nota mínima do SLM (0-10). Valores maiores = mais qualidade, mais perturbações
- **variance_threshold**: Variância máxima aceitável (0-1). Valores menores = mais estabilidade, mais perturbações

### Ajustar Níveis de Perturbação

```python
perturbation_levels=[5, 10, 25, 50, 100, 250, 500, 1000]
```

Adicione/remova valores conforme necessário. Testes iniciarão do menor valor.

### Mudar Modelo de Classificação

Por padrão usa XGBoost. Para usar outro:

```python
# Em model_trainer.py
from sklearn.ensemble import RandomForestClassifier

def train_model(self, X_train, y_train):
    self.model = RandomForestClassifier(
        n_estimators=100,
        random_state=self.random_state
    )
    self.model.fit(X_train, y_train)
```

**Atenção:** O modelo deve ter método `predict_proba()` para LIME funcionar.

### Usar SLMs Diferentes

O projeto suporta qualquer endpoint compatível com OpenAI API. Para usar modelos maiores (ex: Llama, Mistral):

1. Execute seu modelo com compatibilidade OpenAI (ex: via Ollama, LM Studio, vLLM)
2. Execute o projeto apontando para seu servidor:

```bash
python main.py --modo demo --slm-url "http://localhost:11434"
```

## 📚 Estrutura do Código

```
src/
├── model_trainer.py          # Treinamento XGBoost
├── slm_interface.py          # Cliente para SLMs via API
├── explainer_wrapper.py      # Wrappers LIME/SHAP com tracking
├── adaptive_selector.py      # Task 1.1: Seleção adaptativa
├── explanation_aggregator.py # Task 1.2: Agregação de explicações
└── metrics.py                # Métricas de coerência e custo
```

**Fluxo de execução:**
1. `model_trainer.py` carrega dados → treina XGBoost → avalia performance
2. `explainer_wrapper.py` gera explicações LIME/SHAP para instâncias de teste
3. `slm_interface.py` converte explicações técnicas para linguagem natural
4. `adaptive_selector.py` encontra número ótimo de perturbações (Task 1.1)
5. `explanation_aggregator.py` combina N explicações fracas (Task 1.2)
6. `metrics.py` calcula coerência, alinhamento e custos
7. `main.py` orquestra tudo e gera visualizações

## 🎓 Metodologia

Baseado no paper **"Enhancing the Interpretability of SHAP Values Using Large Language Models"** ([arXiv:2409.00079](https://arxiv.org/pdf/2409.00079)):

1. **Modelo Base:** XGBoost treinado em credit risk (risco de crédito)
2. **Explanations:** LIME (perturbações locais) e SHAP (valores de Shapley)
3. **Naturalização:** SLM converte features técnicas em texto compreensível
4. **Avaliação:** SLM auto-avalia coerência (0-10) + variância de estabilidade
5. **Adaptação:** Algoritmo encontra mínimo de perturbações que atinge threshold
6. **Agregação:** Combina N explicações fracas em 1 forte, economizando custo

**Hipóteses testadas:**
- H1: É possível reduzir perturbações em 30-50% mantendo qualidade
- H2: Explicações agregadas (N×10) são comparáveis a fortes (1×500) com 30-40% menos custo

## 📖 Próximos Passos

1. **Explorar Resultados:** Abra os PNGs e CSVs em `outputs/`
2. **Ajustar Parâmetros:** Experimente diferentes thresholds e estratégias
3. **Testar Datasets:** Use seus próprios dados de classificação binária
4. **Comparar Modelos:** Teste RandomForest, LightGBM, etc.
5. **Escalar:** Aumente amostras para análise estatisticamente robusta

## 💡 Dicas de Uso

- **Primeira vez:** Use `--modo demo` para entender o output
- **Pesquisa:** Use `--modo completo` com seed fixo para reprodutibilidade
- **Debug:** Use `--modo sem-slm` para isolar problemas do modelo vs SLM
- **Performance:** Máquinas com GPU podem acelerar XGBoost (configure `tree_method='gpu_hist'`)

## ❓ Suporte

Para dúvidas sobre:
- **Instalação:** Revise esta documentação e teste `test_setup.py`
- **Resultados:** Consulte `summary_report.txt` para interpretação
- **Metodologia:** Leia o paper de referência (pasta `instrucoes/`)
- **Código:** Todos os módulos têm docstrings em português

---

**Desenvolvido para pesquisa em Explicabilidade Adaptativa | 2025**
