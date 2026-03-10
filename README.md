# Explicabilidade Adaptativa com LIME/SHAP e SLMs Locais

Implementação completa de métodos de explicabilidade adaptativa para modelos de machine learning, aprimorados por Small Language Models (SLMs). Este projeto implementa a metodologia de "Enhancing the Interpretability of SHAP Values Using Large Language Models" (arxiv.org/pdf/2409.00079).

## 🎯 Visão Geral do Projeto

Este projeto aborda duas questões de pesquisa principais:

1. **Task 1.1: Seleção Adaptativa de Perturbações**
   - Podemos selecionar adaptativamente o número ótimo de perturbações LIME para cada instância?
   - Objetivo: Minimizar custo computacional mantendo coerência da explicação

2. **Task 1.2: Agregação de Explicações**
   - Podemos combinar múltiplas explicações "fracas" (baixo custo) em uma explicação "forte"?
   - Objetivo: Alcançar qualidade comparável a explicações caras com custo reduzido

## 📊 Dataset

Dataset de Previsão de Risco de Crédito com features incluindo:
- Informações pessoais (idade, renda, tempo de emprego)
- Detalhes do empréstimo (valor, taxa de juros, propósito)
- Histórico de crédito
- Alvo: previsão de inadimplência de empréstimo (classificação binária)

## 🏗️ Arquitetura

```
adaptive-explainability/
├── data/                          # Dataset
│   └── credit_risk_dataset.csv
├── src/                           # Módulos principais
│   ├── model_trainer.py           # Treinamento e avaliação XGBoost
│   ├── slm_interface.py           # Cliente API do Docker SLM
│   ├── explainer_wrapper.py       # LIME/SHAP com rastreamento
│   ├── adaptive_selector.py       # Implementação Task 1.1
│   ├── explanation_aggregator.py  # Implementação Task 1.2
│   └── metrics.py                 # Métricas de coerência e custo
├── configs/                       # Scripts de configuração
│   ├── docker_setup.sh            # Setup Bash
│   └── docker_setup.ps1           # Setup PowerShell
├── outputs/                       # Relatórios e gráficos gerados
├── instrucoes/                    # Papers de referência
├── main.py                        # Script principal de execução
└── requirements.txt               # Dependências Python
```

## 🚀 Início Rápido

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar Docker SLMs

**Windows (PowerShell):**
```powershell
cd configs
.\docker_setup.ps1
```

**Linux/Mac:**
```bash
cd configs
bash docker_setup.sh
```

Isso irá:
- Baixar modelos Qwen2.5 e IBM Granite 4.0 Nano
- Iniciar containers nas portas 8080 (primário) e 8081 (backup)
- Testar conectividade

### 3. Verificar Configuração

```bash
docker ps
# Deve mostrar qwen2.5-server e granite-nano-server rodando

# Testar API
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Olá!", "max_tokens": 10}'
```

### 4. Executar Análise

**Demo rápido (5-10 minutos):**
```bash
python main.py --modo demo
```

**Análise completa (1-2 horas):**
```bash
python main.py --modo completo
```

**Apenas modelo (sem SLM):**
```bash
python main.py --modo sem-slm
```

## 📈 Funcionalidades Principais

### Seleção Adaptativa de Perturbações
- **Busca sequencial**: Testa níveis de perturbação do baixo ao alto
- **Busca binária**: Mais rápida porém menos precisa
- **Métricas de coerência**: 
  - Auto-avaliação do SLM (escala 0-10)
  - Variância de estabilidade das features
  - Score de coerência composto
- **Características da instância**: Correlaciona complexidade com perturbações ótimas

### Agregação de Explicações
- **Estratégia de concatenação**: Combinação simples
- **Estratégia de síntese**: Agregação guiada com detecção de consenso
- **Métricas de qualidade**:
  - Similaridade de Jaccard (alinhamento de features)
  - Concordância direcional
  - Correlação de importância
- **Análise de custo**: Compara N×fracas vs 1×forte

### Visualizações
- Dashboards de exploração de dados
- Performance do modelo (curvas ROC, PR, matriz de confusão)
- Rankings de importância das features
- Curvas perturbação vs coerência
- Gráficos de trade-off custo-qualidade
- Comparações LIME vs SHAP

## 🔬 Metodologia

Baseado no paper "Enhancing the Interpretability of SHAP Values Using Large Language Models":

1. **Treinar modelo preditivo** (XGBoost para risco de crédito)
2. **Gerar explicações** usando LIME/SHAP
3. **Converter para linguagem natural** via SLM local
4. **Avaliar coerência** usando auto-avaliação do SLM + estabilidade
5. **Selecionar adaptativamente** o mínimo de perturbações que atendem o threshold
6. **Agregar explicações fracas** para eficiência de custo

## 📊 Resultados Esperados

**Task 1.1:**
- Redução de 30-50% nas perturbações em média
- Instâncias próximas à fronteira de decisão requerem mais perturbações
- Predições de alta confiança usam menos perturbações

**Task 1.2:**
- Alinhamento de features 70%+ com ground truth
- Economia de 30-40% no custo computacional
- Estratégia de síntese supera concatenação

## 🛠️ Configuração

### Parâmetros do Modelo
Editar em main.py ou módulos:
- `RANDOM_STATE = 42`
- `WEAK_PERTURBATIONS = 10`
- `N_WEAK_EXPLANATIONS = 20`
- `STRONG_PERTURBATIONS = 500`

### Seletor Adaptativo
```python
AdaptivePerturbationSelector(
    perturbation_levels=[5, 10, 25, 50, 100, 250, 500, 1000],
    coherence_threshold=7.0,        # Escala 0-10
    variance_threshold=0.15,        # Variância máxima aceitável
    stability_runs=3                # Execuções para teste de estabilidade
)
```

### Tamanhos de Amostra
- `n_test_instances`: Número de instâncias para analisar (50 para demo, 200+ para completo)
- `n_adaptive_samples`: Instâncias para Task 1.1 (5 para demo)
- `n_aggregation_samples`: Instâncias para Task 1.2 (5 para demo)

## 📦 Arquivos de Saída

Todas as saídas são salvas em `outputs/`:
- `data_exploration.png` - Visualizações do dataset
- `model_performance.png` - Métricas do classificador
- `feature_importance.png` - Rankings de features XGBoost
- `task1.1_adaptive_selection.png` - Resultados da seleção adaptativa
- `task1.1_characteristics_correlation.png` - Características das instâncias
- `task1.2_aggregation_results.png` - Comparação de agregação
- `task1.2_optimal_n.png` - Número ótimo de explicações fracas
- `lime_vs_shap_comparison.png` - Comparação de métodos
- `*.csv` - Tabelas de resultados detalhados
- `summary_report.txt` - Resumo textual

## 🐛 Solução de Problemas

### SLM Não Disponível
```
⚠ SLM not available: Connection refused
```
**Solução:** 
1. Execute `docker ps` para verificar os containers
2. Reinicie: `docker restart qwen2.5-server`
3. Verifique logs: `docker logs qwen2.5-server`

### Falta de Memória
```
Docker error: Out of memory
```
**Solução:** Aumente o limite de memória do Docker nas configurações do Docker Desktop (recomendado 8GB+)

### Performance Lenta
**Soluções:**
1. Reduza os tamanhos de amostra em main.py
2. Use busca binária ao invés de sequencial
3. Reduza `stability_runs` para 2
4. Use níveis menores de perturbação

### Erros LIME/SHAP
```
TypeError: predict_proba() missing
```
**Solução:** Certifique-se que o modelo tem o método `predict_proba` (XGBoost, classificadores sklearn)

## 📚 Referências

1. [Enhancing the Interpretability of SHAP Values Using Large Language Models](https://arxiv.org/pdf/2409.00079)
2. [Paper LIME](https://arxiv.org/abs/1602.04938)
3. [Paper SHAP](https://arxiv.org/abs/1705.07874)
4. [Docker Model Runner](https://hub.docker.com/r/ai/)

## 🔄 Estendendo o Projeto

### Adicionar Novos Datasets
1. Coloque o CSV em `data/`
2. Atualize `data_path` em main.py
3. Ajuste pré-processamento se necessário

### Experimentar Diferentes Modelos
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(...)
trainer.model = model
```

### Adicionar Métricas Personalizadas
Estenda `metrics.py`:
```python
@staticmethod
def custom_metric(explanation, ground_truth):
    # Sua implementação de métrica
    pass
```

### Test Other SLMs
Update `slm_interface.py` URLs or add new endpoints

## 👥 Contributors

Project developed for adaptive explainability research using:
- Python 3.9+
- XGBoost 2.0+
- LIME 0.2+
- SHAP 0.42+
- Qwen2.5 / IBM Granite 4.0 Nano

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

Based on methodology from "Enhancing the Interpretability of SHAP Values Using Large Language Models" and XAI best practices.

---

**Questions or Issues?** Check the troubleshooting section or review the detailed comments in the notebook.
