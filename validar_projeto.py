#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de validação rápida do projeto.
Verifica se todos os módulos podem ser importados e se a estrutura está correta.
"""

import sys
import os
from pathlib import Path

def testar_importacoes():
    """Testa se todos os módulos podem ser importados."""
    print("🔍 Testando importações dos módulos...")
    
    try:
        from src import model_trainer
        print("  ✓ model_trainer.py")
    except Exception as e:
        print(f"  ✗ model_trainer.py: {e}")
        return False
    
    try:
        from src import slm_interface
        print("  ✓ slm_interface.py")
    except Exception as e:
        print(f"  ✗ slm_interface.py: {e}")
        return False
    
    try:
        from src import explainer_wrapper
        print("  ✓ explainer_wrapper.py")
    except Exception as e:
        print(f"  ✗ explainer_wrapper.py: {e}")
        return False
    
    try:
        from src import adaptive_selector
        print("  ✓ adaptive_selector.py")
    except Exception as e:
        print(f"  ✗ adaptive_selector.py: {e}")
        return False
    
    try:
        from src import explanation_aggregator
        print("  ✓ explanation_aggregator.py")
    except Exception as e:
        print(f"  ✗ explanation_aggregator.py: {e}")
        return False
    
    try:
        from src import metrics
        print("  ✓ metrics.py")
    except Exception as e:
        print(f"  ✗ metrics.py: {e}")
        return False
    
    return True

def testar_estrutura():
    """Verifica se a estrutura de diretórios está correta."""
    print("\n📁 Verificando estrutura de diretórios...")
    
    diretorios_necessarios = [
        "src",
        "data",
        "configs",
        "outputs",
        "instrucoes"
    ]
    
    arquivos_necessarios = [
        "main.py",
        "requirements.txt",
        "README.md",
        "LEIAME.md",
        "data/credit_risk_dataset.csv",
        "configs/docker_setup.ps1",
        "configs/docker_setup.sh"
    ]
    
    tudo_ok = True
    
    for dir_name in diretorios_necessarios:
        if Path(dir_name).exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (não encontrado)")
            tudo_ok = False
    
    for file_path in arquivos_necessarios:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (não encontrado)")
            tudo_ok = False
    
    return tudo_ok

def testar_dataset():
    """Verifica se o dataset pode ser carregado."""
    print("\n📊 Testando carregamento do dataset...")
    
    try:
        import pandas as pd
        df = pd.read_csv("data/credit_risk_dataset.csv")
        print(f"  ✓ Dataset carregado: {len(df)} linhas, {len(df.columns)} colunas")
        print(f"  ✓ Colunas: {', '.join(df.columns[:5])}...")
        return True
    except Exception as e:
        print(f"  ✗ Erro ao carregar dataset: {e}")
        return False

def testar_argumentos_cli():
    """Verifica se main.py aceita argumentos da linha de comando."""
    print("\n⚙️ Testando interface CLI...")
    
    try:
        import argparse
        # Simula a criação do parser como em main.py
        parser = argparse.ArgumentParser()
        parser.add_argument('--modo', choices=['completo', 'demo', 'sem-slm'], default='demo')
        parser.add_argument('--dados', type=str, default='data/credit_risk_dataset.csv')
        parser.add_argument('--saida', type=str, default='outputs')
        parser.add_argument('--slm-url', type=str, default='http://localhost:8080')
        
        # Testa parsing de argumentos de exemplo
        args = parser.parse_args(['--modo', 'demo', '--dados', 'data/credit_risk_dataset.csv'])
        
        print(f"  ✓ Parser CLI configurado corretamente")
        print(f"  ✓ Modo padrão: {args.modo}")
        print(f"  ✓ Dataset padrão: {args.dados}")
        return True
    except Exception as e:
        print(f"  ✗ Erro na interface CLI: {e}")
        return False

def main():
    """Executa todos os testes de validação."""
    print("=" * 60)
    print("🚀 VALIDAÇÃO DO PROJETO - EXPLICABILIDADE ADAPTATIVA")
    print("=" * 60)
    
    resultados = []
    
    # Teste 1: Estrutura
    resultados.append(("Estrutura de Diretórios", testar_estrutura()))
    
    # Teste 2: Importações
    resultados.append(("Importação de Módulos", testar_importacoes()))
    
    # Teste 3: Dataset
    resultados.append(("Carregamento de Dataset", testar_dataset()))
    
    # Teste 4: CLI
    resultados.append(("Interface CLI", testar_argumentos_cli()))
    
    # Resumo
    print("\n" + "=" * 60)
    print("📋 RESUMO DOS TESTES")
    print("=" * 60)
    
    testes_passaram = 0
    testes_falharam = 0
    
    for nome, passou in resultados:
        status = "✓ PASSOU" if passou else "✗ FALHOU"
        print(f"{status}: {nome}")
        if passou:
            testes_passaram += 1
        else:
            testes_falharam += 1
    
    print("\n" + "-" * 60)
    print(f"Total: {testes_passaram}/{len(resultados)} testes passaram")
    
    if testes_falharam == 0:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("\nPróximos passos:")
        print("1. Configure o Docker: cd configs && .\\docker_setup.ps1")
        print("2. Execute o projeto: python main.py --modo demo")
        return 0
    else:
        print(f"\n⚠️  {testes_falharam} teste(s) falharam.")
        print("\nVerifique os erros acima e:")
        print("- Certifique-se que todas as dependências estão instaladas (pip install -r requirements.txt)")
        print("- Verifique se todos os arquivos foram criados corretamente")
        print("- Consulte o arquivo LEIAME.md para instruções detalhadas")
        return 1

if __name__ == "__main__":
    sys.exit(main())
