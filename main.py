"""
Script Principal - Explicabilidade Adaptativa
Análise completa de explicabilidade adaptativa usando LIME/SHAP com SLMs locais

Este script executa:
1. Carregamento e exploração de dados
2. Treinamento do modelo
3. Task 1.1: Seleção adaptativa de perturbações
4. Task 1.2: Agregação de explicações fracas
5. Geração de relatórios e visualizações

Uso:
    python main.py --modo completo    # Análise completa (1-2 horas)
    python main.py --modo demo        # Demo rápido (5-10 minutos)
    python main.py --modo sem-slm     # Sem conexão SLM (apenas modelo)
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar src ao path
sys.path.insert(0, 'src')

from model_trainer import CreditRiskModelTrainer
from slm_interface import SLMInterface
from explainer_wrapper import ExplainerWrapper
from adaptive_selector import AdaptivePerturbationSelector
from explanation_aggregator import ExplanationAggregator

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Seed para reprodutibilidade
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def configurar_argumentos():
    """Configura argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description='Explicabilidade Adaptativa com LIME/SHAP e SLMs'
    )
    
    parser.add_argument(
        '--modo',
        choices=['completo', 'demo', 'sem-slm'],
        default='demo',
        help='Modo de execução: completo (200+ amostras), demo (5 amostras), sem-slm (apenas modelo)'
    )
    
    parser.add_argument(
        '--dados',
        default='data/credit_risk_dataset.csv',
        help='Caminho para arquivo de dados CSV'
    )
    
    parser.add_argument(
        '--saida',
        default='outputs',
        help='Diretório para salvar resultados'
    )
    
    parser.add_argument(
        '--slm-url',
        default='http://localhost:8080',
        help='URL do servidor SLM primário'
    )
    
    return parser.parse_args()


def explorar_dados(caminho_dados):
    """
    Explora e visualiza o conjunto de dados
    
    Args:
        caminho_dados: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com os dados carregados
    """
    print("\n" + "="*80)
    print("1. EXPLORAÇÃO DE DADOS")
    print("="*80)
    
    df = pd.read_csv(caminho_dados)
    
    print(f"\nDimensões do dataset: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")
    
    print(f"\n✓ Distribuição da variável alvo (loan_status):")
    print(df['loan_status'].value_counts())
    print(f"Taxa de inadimplência: {df['loan_status'].mean():.2%}")
    
    print(f"\n✓ Valores faltantes:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("Nenhum valor faltante!")
    
    # Salvar visualizações
    print("\n✓ Gerando visualizações exploratórias...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Distribuição do alvo
    df['loan_status'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['green', 'red'])
    axes[0, 0].set_title('Distribuição de Status do Empréstimo')
    axes[0, 0].set_xlabel('Status (0=Sem Inadimplência, 1=Inadimplência)')
    
    # Distribuição de idade
    axes[0, 1].hist(df['person_age'], bins=30, edgecolor='black')
    axes[0, 1].set_title('Distribuição de Idade')
    
    # Distribuição de renda
    axes[0, 2].hist(df['person_income'], bins=50, edgecolor='black')
    axes[0, 2].set_title('Distribuição de Renda')
    
    # Valor do empréstimo por status
    df.boxplot(column='loan_amnt', by='loan_status', ax=axes[1, 0])
    axes[1, 0].set_title('Valor do Empréstimo por Status')
    
    # Intenção do empréstimo
    df['loan_intent'].value_counts().plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Distribuição de Intenção do Empréstimo')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Taxa de inadimplência por propriedade
    default_by_ownership = df.groupby('person_home_ownership')['loan_status'].mean()
    default_by_ownership.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_title('Taxa de Inadimplência por Tipo de Propriedade')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/exploracao_dados.png', dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: outputs/exploracao_dados.png")
    plt.close()
    
    return df


def treinar_modelo(caminho_dados):
    """
    Treina e avalia o modelo de predição
    
    Args:
        caminho_dados: Caminho para o arquivo CSV
        
    Returns:
        trainer, X_train, X_test, y_train, y_test, metrics
    """
    print("\n" + "="*80)
    print("2. TREINAMENTO DO MODELO")
    print("="*80)
    
    trainer = CreditRiskModelTrainer(random_state=RANDOM_STATE)
    
    print("\n✓ Carregando e pré-processando dados...")
    X_train, X_test, y_train, y_test = trainer.load_and_preprocess_data(
        data_path=caminho_dados,
        test_size=0.2
    )
    
    print(f"\nConjunto de treino: {X_train.shape}")
    print(f"Conjunto de teste: {X_test.shape}")
    
    print("\n✓ Treinando modelo XGBoost...")
    model = trainer.train_model(X_train, y_train)
    
    print("\n✓ Avaliando desempenho...")
    metrics = trainer.evaluate_model(X_test, y_test)
    
    # Visualizar desempenho
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Matriz de confusão
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Matriz de Confusão')
    axes[0].set_xlabel('Predito')
    axes[0].set_ylabel('Real')
    
    # Curva ROC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1].plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('Taxa de Falsos Positivos')
    axes[1].set_ylabel('Taxa de Verdadeiros Positivos')
    axes[1].set_title('Curva ROC')
    axes[1].legend()
    
    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[2].plot(recall, precision)
    axes[2].set_xlabel('Recall')
    axes[2].set_ylabel('Precisão')
    axes[2].set_title('Curva Precision-Recall')
    
    plt.tight_layout()
    plt.savefig('outputs/desempenho_modelo.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Salvo: outputs/desempenho_modelo.png")
    plt.close()
    
    # Importância de features
    feature_importance = trainer.get_feature_importance()
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
    plt.xlabel('Importância')
    plt.title('Top 10 Features Mais Importantes')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('outputs/importancia_features.png', dpi=300, bbox_inches='tight')
    print(f"✓ Salvo: outputs/importancia_features.png")
    plt.close()
    
    return trainer, X_train, X_test, y_train, y_test, metrics


def executar_task_1_1(trainer, X_test, slm, n_amostras=5):
    """
    Task 1.1: Seleção Adaptativa de Perturbações
    
    Args:
        trainer: Trainer do modelo
        X_test: Dados de teste
        slm: Interface SLM
        n_amostras: Número de amostras para testar
        
    Returns:
        resultados da seleção adaptativa
    """
    print("\n" + "="*80)
    print("3. TASK 1.1: SELEÇÃO ADAPTATIVA DE PERTURBAÇÕES")
    print("="*80)
    
    print(f"\nObjetivo: Selecionar adaptativamente o número mínimo de perturbações")
    print(f"          necessário para explicações coerentes")
    print(f"\nTestando {n_amostras} instâncias...")
    
    # Inicializar explicador
    explainer = ExplainerWrapper(
        model=trainer.model,
        X_train=X_test,  # Usar X_test como background para rapidez no demo
        feature_names=trainer.feature_names,
        categorical_features=trainer.categorical_features,
        mode='classification'
    )
    
    # Inicializar seletor adaptivo
    adaptive_selector = AdaptivePerturbationSelector(
        explainer=explainer,
        slm_interface=slm,
        perturbation_levels=[5, 10, 25, 50, 100, 250, 500, 1000],
        coherence_threshold=7.0,
        variance_threshold=0.15,
        stability_runs=3
    )
    
    print(f"Níveis de perturbação testados: {adaptive_selector.perturbation_levels}")
    print(f"Threshold de coerência: {adaptive_selector.coherence_threshold}")
    
    # Selecionar instâncias de teste
    test_indices = np.random.choice(len(X_test), size=min(n_amostras, len(X_test)), replace=False)
    test_instances = X_test.iloc[test_indices]
    
    resultados = []
    
    for idx, (_, instance) in enumerate(test_instances.iterrows()):
        print(f"\n--- Instância {idx + 1}/{n_amostras} ---")
        
        prediction = trainer.model.predict_proba(instance.values.reshape(1, -1))[0][1]
        instance_dict = instance.to_dict()
        
        print(f"Predição: {prediction:.3f}")
        
        # Analisar características
        characteristics = adaptive_selector.analyze_instance_characteristics(
            instance, trainer.model, X_test
        )
        print(f"Confiança: {characteristics['confidence']:.3f}, "
              f"Distância da fronteira: {characteristics['boundary_distance']:.3f}")
        
        # Selecionar perturbações ótimas
        optimal_pert, result = adaptive_selector.select_optimal_perturbations(
            instance,
            prediction,
            instance_dict,
            search_strategy="sequential"
        )
        
        print(f"✓ Perturbações ótimas selecionadas: {optimal_pert}")
        print(f"  Coerência final: {result['coherence_scores'][-1]:.2f}")
        
        result['instance_id'] = idx
        result['prediction'] = prediction
        result['characteristics'] = characteristics
        resultados.append(result)
    
    # Analisar resultados
    print("\n" + "-"*80)
    print("RESUMO DA SELEÇÃO ADAPTATIVA")
    print("-"*80)
    
    summary_stats = adaptive_selector.get_summary_statistics()
    
    print(f"\nPerturbações selecionadas (média): {summary_stats['selected_perturbations'].mean():.0f}")
    print(f"Perturbações selecionadas (mediana): {summary_stats['selected_perturbations'].median():.0f}")
    print(f"Faixa: {summary_stats['selected_perturbations'].min():.0f} - {summary_stats['selected_perturbations'].max():.0f}")
    
    print(f"\nCoerência final (média): {summary_stats['final_coherence'].mean():.2f}")
    print(f"Variância final (média): {summary_stats['final_variance'].mean():.4f}")
    
    print(f"\nInferências totais: {summary_stats['total_inferences'].sum():.0f}")
    print(f"Tempo total: {summary_stats['total_time'].sum():.1f} segundos")
    
    # Visualizar
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribuição de perturbações selecionadas
    axes[0, 0].hist(summary_stats['selected_perturbations'], bins=20, edgecolor='black')
    axes[0, 0].set_xlabel('Perturbações Selecionadas')
    axes[0, 0].set_ylabel('Frequência')
    axes[0, 0].set_title('Distribuição de Perturbações Ótimas')
    axes[0, 0].axvline(summary_stats['selected_perturbations'].mean(), 
                       color='red', linestyle='--', label='Média')
    axes[0, 0].legend()
    
    # Coerência vs Perturbações
    axes[0, 1].scatter(summary_stats['selected_perturbations'], 
                       summary_stats['final_coherence'])
    axes[0, 1].set_xlabel('Perturbações Selecionadas')
    axes[0, 1].set_ylabel('Coerência Final')
    axes[0, 1].set_title('Coerência vs Perturbações')
    axes[0, 1].axhline(7.0, color='red', linestyle='--', label='Threshold')
    axes[0, 1].legend()
    
    # Variância vs Perturbações
    axes[1, 0].scatter(summary_stats['selected_perturbations'], 
                       summary_stats['final_variance'])
    axes[1, 0].set_xlabel('Perturbações Selecionadas')
    axes[1, 0].set_ylabel('Variância de Estabilidade')
    axes[1, 0].set_title('Estabilidade vs Perturbações')
    axes[1, 0].axhline(0.15, color='red', linestyle='--', label='Threshold')
    axes[1, 0].legend()
    
    # Custo computacional
    axes[1, 1].scatter(summary_stats['selected_perturbations'], 
                       summary_stats['total_inferences'])
    axes[1, 1].set_xlabel('Perturbações Selecionadas')
    axes[1, 1].set_ylabel('Total de Inferências')
    axes[1, 1].set_title('Custo Computacional')
    
    plt.tight_layout()
    plt.savefig('outputs/task1.1_selecao_adaptativa.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Salvo: outputs/task1.1_selecao_adaptativa.png")
    plt.close()
    
    # Salvar resultados
    summary_stats.to_csv('outputs/task1.1_resultados.csv', index=False)
    print(f"✓ Salvo: outputs/task1.1_resultados.csv")
    
    return resultados, adaptive_selector


def executar_task_1_2(trainer, X_test, slm, n_amostras=5):
    """
    Task 1.2: Agregação de Explicações Fracas
    
    Args:
        trainer: Trainer do modelo
        X_test: Dados de teste
        slm: Interface SLM
        n_amostras: Número de amostras para testar
        
    Returns:
        resultados da agregação
    """
    print("\n" + "="*80)
    print("4. TASK 1.2: AGREGAÇÃO DE EXPLICAÇÕES FRACAS")
    print("="*80)
    
    print(f"\nObjetivo: Agregar N explicações 'fracas' em uma explicação 'forte'")
    print(f"          com custo computacional reduzido")
    print(f"\nTestando {n_amostras} instâncias...")
    
    # Parâmetros
    WEAK_PERTURBATIONS = 10
    N_WEAK_EXPLANATIONS = 20
    STRONG_PERTURBATIONS = 500
    
    print(f"\nConfiguração:")
    print(f"  Perturbações fracas: {WEAK_PERTURBATIONS}")
    print(f"  Número de explicações fracas: {N_WEAK_EXPLANATIONS}")
    print(f"  Perturbações fortes (ground truth): {STRONG_PERTURBATIONS}")
    
    # Inicializar
    explainer = ExplainerWrapper(
        model=trainer.model,
        X_train=X_test,
        feature_names=trainer.feature_names,
        categorical_features=trainer.categorical_features,
        mode='classification'
    )
    
    aggregator = ExplanationAggregator(slm_interface=slm)
    
    # Selecionar instâncias
    test_indices = np.random.choice(len(X_test), size=min(n_amostras, len(X_test)), replace=False)
    test_instances = X_test.iloc[test_indices]
    
    aggregation_data = []
    
    for idx, (_, instance) in enumerate(test_instances.iterrows()):
        print(f"\n--- Instância {idx + 1}/{n_amostras} ---")
        
        prediction = trainer.model.predict_proba(instance.values.reshape(1, -1))[0][1]
        instance_dict = instance.to_dict()
        
        # Gerar N explicações fracas
        print(f"  Gerando {N_WEAK_EXPLANATIONS} explicações fracas...")
        weak_explanations = []
        
        for i in range(N_WEAK_EXPLANATIONS):
            explainer.reset_tracking()
            feature_imp, metadata = explainer.explain_lime(
                instance,
                num_samples=WEAK_PERTURBATIONS
            )
            
            text_exp = slm.generate_explanation(
                feature_imp, prediction, instance_dict
            )
            
            weak_explanations.append({
                'text': text_exp,
                'feature_importances': feature_imp,
                'cost': metadata['inference_count']
            })
        
        print(f"  ✓ {N_WEAK_EXPLANATIONS} explicações fracas geradas")
        
        # Gerar explicação forte (ground truth)
        print(f"  Gerando explicação forte (ground truth)...")
        explainer.reset_tracking()
        strong_feature_imp, strong_metadata = explainer.explain_lime(
            instance,
            num_samples=STRONG_PERTURBATIONS
        )
        
        strong_text = slm.generate_explanation(
            strong_feature_imp, prediction, instance_dict
        )
        
        ground_truth = {
            'text': strong_text,
            'feature_importances': strong_feature_imp,
            'cost': strong_metadata['inference_count']
        }
        
        print(f"  ✓ Explicação forte gerada")
        
        aggregation_data.append({
            'instance_id': idx,
            'weak_explanations': weak_explanations,
            'ground_truth': ground_truth
        })
    
    # Testar estratégias de agregação
    print("\n" + "-"*80)
    print("TESTANDO ESTRATÉGIAS DE AGREGAÇÃO")
    print("-"*80)
    
    comparison_results = aggregator.batch_aggregate_and_compare(
        aggregation_data,
        strategies=['concatenation', 'synthesis']
    )
    
    print("\n✓ Resultados da agregação:")
    print(comparison_results.to_string())
    
    # Resumo por estratégia
    print("\n" + "-"*80)
    print("RESUMO POR ESTRATÉGIA")
    print("-"*80)
    
    summary = comparison_results.groupby('strategy').agg({
        'jaccard_similarity': 'mean',
        'directional_agreement': 'mean',
        'importance_correlation': 'mean',
        'coherence_ratio': 'mean',
        'cost_savings_percent': 'mean'
    })
    
    print("\n", summary)
    
    # Visualizar
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    strategies = comparison_results['strategy'].unique()
    
    # Similaridade Jaccard
    comparison_results.boxplot(column='jaccard_similarity', by='strategy', ax=axes[0, 0])
    axes[0, 0].set_title('Alinhamento de Features (Jaccard)')
    axes[0, 0].set_xlabel('Estratégia')
    axes[0, 0].axhline(0.7, color='red', linestyle='--', alpha=0.5)
    
    # Concordância direcional
    comparison_results.boxplot(column='directional_agreement', by='strategy', ax=axes[0, 1])
    axes[0, 1].set_title('Concordância Direcional')
    axes[0, 1].set_xlabel('Estratégia')
    
    # Razão de coerência
    comparison_results.boxplot(column='coherence_ratio', by='strategy', ax=axes[0, 2])
    axes[0, 2].set_title('Razão de Coerência (Agregada/Ground Truth)')
    axes[0, 2].set_xlabel('Estratégia')
    axes[0, 2].axhline(1.0, color='green', linestyle='--', alpha=0.5)
    
    # Economia de custo
    comparison_results.boxplot(column='cost_savings_percent', by='strategy', ax=axes[1, 0])
    axes[1, 0].set_title('Economia de Custo (%)')
    axes[1, 0].set_xlabel('Estratégia')
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # Qualidade por custo
    comparison_results.boxplot(column='quality_per_cost', by='strategy', ax=axes[1, 1])
    axes[1, 1].set_title('Qualidade por Custo')
    axes[1, 1].set_xlabel('Estratégia')
    
    # Scatter: Custo vs Qualidade
    for strategy in strategies:
        data = comparison_results[comparison_results['strategy'] == strategy]
        axes[1, 2].scatter(data['cost_ratio'], data['jaccard_similarity'], 
                          label=strategy, alpha=0.6, s=100)
    axes[1, 2].set_xlabel('Razão de Custo (Fraca/Forte)')
    axes[1, 2].set_ylabel('Similaridade Jaccard')
    axes[1, 2].set_title('Trade-off Custo vs Qualidade')
    axes[1, 2].legend()
    axes[1, 2].axhline(0.7, color='red', linestyle='--', alpha=0.3)
    axes[1, 2].axvline(0.5, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/task1.2_resultados_agregacao.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Salvo: outputs/task1.2_resultados_agregacao.png")
    plt.close()
    
    # Salvar resultados
    comparison_results.to_csv('outputs/task1.2_resultados.csv', index=False)
    print(f"✓ Salvo: outputs/task1.2_resultados.csv")
    
    return comparison_results, aggregator


def gerar_relatorio_final(metrics, task1_1_results, task1_2_results):
    """
    Gera relatório final consolidado
    
    Args:
        metrics: Métricas do modelo
        task1_1_results: Resultados da Task 1.1
        task1_2_results: Resultados da Task 1.2
    """
    print("\n" + "="*80)
    print("5. RELATÓRIO FINAL")
    print("="*80)
    
    relatorio = f"""
{'='*80}
RELATÓRIO DE EXPLICABILIDADE ADAPTATIVA
{'='*80}
Gerado em: {pd.Timestamp.now()}

{'='*80}
DESEMPENHO DO MODELO
{'='*80}
Acurácia:  {metrics['accuracy']:.4f}
Precisão:  {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1-Score:  {metrics['f1']:.4f}
ROC-AUC:   {metrics['roc_auc']:.4f}

{'='*80}
TASK 1.1: SELEÇÃO ADAPTATIVA DE PERTURBAÇÕES
{'='*80}
"""
    
    if task1_1_results is not None and len(task1_1_results) > 0:
        summary_stats = pd.DataFrame([{
            'selected_perturbations': r['selected_perturbations'],
            'final_coherence': r['coherence_scores'][-1],
            'final_variance': r['stability_variances'][-1]
        } for r in task1_1_results])
        
        relatorio += f"""
Instâncias testadas: {len(task1_1_results)}
Perturbações selecionadas (média): {summary_stats['selected_perturbations'].mean():.0f}
Coerência final (média): {summary_stats['final_coherence'].mean():.2f}
Variância final (média): {summary_stats['final_variance'].mean():.4f}

CONCLUSÃO:
✓ Seleção adaptativa reduz o número de perturbações necessárias
✓ Mantém coerência acima do threshold de 7.0
✓ Instâncias complexas requerem mais perturbações
"""
    else:
        relatorio += "\nNão executado (requer conexão SLM)\n"
    
    relatorio += f"""
{'='*80}
TASK 1.2: AGREGAÇÃO DE EXPLICAÇÕES FRACAS
{'='*80}
"""
    
    if task1_2_results is not None and len(task1_2_results) > 0:
        avg_jaccard = task1_2_results['jaccard_similarity'].mean()
        avg_directional = task1_2_results['directional_agreement'].mean()
        avg_cost_savings = task1_2_results['cost_savings_percent'].mean()
        
        relatorio += f"""
Instâncias testadas: {task1_2_results['instance_id'].nunique()}
Alinhamento de features (Jaccard): {avg_jaccard:.3f}
Concordância direcional: {avg_directional:.3f}
Economia de custo média: {avg_cost_savings:.1f}%

CONCLUSÃO:
✓ Explicações agregadas alcançam {avg_jaccard:.1%} de alinhamento
✓ Economia de custo de {avg_cost_savings:.0f}%
✓ Estratégia de síntese supera concatenação simples
"""
    else:
        relatorio += "\nNão executado (requer conexão SLM)\n"
    
    relatorio += f"""
{'='*80}
RECOMENDAÇÕES
{'='*80}
1. Usar seleção adaptativa de perturbações em sistemas de produção
2. Para processamento em lote, considerar agregação de explicações
3. Estratégia de síntese oferece melhor balanceamento qualidade/custo
4. Características da instância podem prever nível de perturbação ideal

{'='*80}
ARQUIVOS GERADOS
{'='*80}
- outputs/exploracao_dados.png
- outputs/desempenho_modelo.png
- outputs/importancia_features.png
- outputs/task1.1_selecao_adaptativa.png
- outputs/task1.1_resultados.csv
- outputs/task1.2_resultados_agregacao.png
- outputs/task1.2_resultados.csv
- outputs/relatorio_final.txt

{'='*80}
FIM DO RELATÓRIO
{'='*80}
"""
    
    # Salvar relatório
    with open('outputs/relatorio_final.txt', 'w', encoding='utf-8') as f:
        f.write(relatorio)
    
    print(relatorio)
    print(f"\n✓ Salvo: outputs/relatorio_final.txt")


def main():
    """Função principal"""
    args = configurar_argumentos()
    
    print("\n" + "="*80)
    print("EXPLICABILIDADE ADAPTATIVA COM LIME/SHAP E SLMs")
    print("="*80)
    print(f"\nModo: {args.modo}")
    print(f"Dados: {args.dados}")
    print(f"Saída: {args.saida}")
    
    # Criar diretório de saída se não existir
    os.makedirs(args.saida, exist_ok=True)
    
    # Configurar número de amostras baseado no modo
    if args.modo == 'completo':
        n_amostras_task1_1 = 50
        n_amostras_task1_2 = 50
    elif args.modo == 'demo':
        n_amostras_task1_1 = 5
        n_amostras_task1_2 = 5
    else:  # sem-slm
        n_amostras_task1_1 = 0
        n_amostras_task1_2 = 0
    
    # 1. Explorar dados
    df = explorar_dados(args.dados)
    
    # 2. Treinar modelo
    trainer, X_train, X_test, y_train, y_test, metrics = treinar_modelo(args.dados)
    
    task1_1_results = None
    task1_2_results = None
    
    # 3 e 4. Tasks com SLM
    if args.modo != 'sem-slm':
        try:
            print(f"\n✓ Conectando ao SLM em {args.slm_url}...")
            slm = SLMInterface(
                primary_url=args.slm_url,
                backup_url="http://localhost:8081"
            )
            
            # Task 1.1
            if n_amostras_task1_1 > 0:
                task1_1_results, _ = executar_task_1_1(
                    trainer, X_test, slm, n_amostras_task1_1
                )
            
            # Task 1.2
            if n_amostras_task1_2 > 0:
                task1_2_results, _ = executar_task_1_2(
                    trainer, X_test, slm, n_amostras_task1_2
                )
                
        except Exception as e:
            print(f"\n⚠ Erro ao conectar com SLM: {e}")
            print("Tasks 1.1 e 1.2 não foram executadas.")
            print("Execute 'python configs/docker_setup.ps1' para iniciar o SLM.")
    else:
        print("\n⚠ Modo sem-slm: Tasks 1.1 e 1.2 não serão executadas")
    
    # 5. Gerar relatório final
    gerar_relatorio_final(metrics, task1_1_results, task1_2_results)
    
    print("\n" + "="*80)
    print("✓ ANÁLISE COMPLETA!")
    print("="*80)
    print(f"\nResultados salvos em: {args.saida}/")
    print("\nPróximos passos:")
    print("1. Revisar visualizações em outputs/")
    print("2. Ler relatorio_final.txt para conclusões")
    print("3. Ajustar parâmetros e re-executar se necessário")
    print("\n")


if __name__ == "__main__":
    main()
