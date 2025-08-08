# VetLLM - Assistente IA para Medicina Veterinária

🚧 **Este projeto está em fase de produção e testes** 🚧

## Sobre o Projeto

VetLLM é um assistente de inteligência artificial especializado em medicina veterinária, desenvolvido para auxiliar profissionais veterinários com informações técnicas precisas e baseadas em evidências científicas. O sistema utiliza tecnologias avançadas de recuperação de informações (RAG) combinadas com modelos de linguagem especializados.

## Características Principais

### 🧠 Busca Inteligente com LLM
- Geração automática de termos de busca relevantes usando IA
- Busca multilíngue (português e inglês)
- Filtros automáticos para conteúdo relevante
- Validação inteligente de contexto

### 🔍 Sistema RAG Avançado
- Embeddings multilíngues para melhor compreensão
- Base de dados vetorial com Chroma
- Normalização de scores de relevância
- Remoção automática de duplicatas

### ⚡ Otimização de Performance
- Quantização 4-bit para modelos LLM
- Carregamento otimizado de memória
- Processamento eficiente em GPU/CPU

### 🩺 Especialização Veterinária
- Templates específicos para medicina veterinária
- Linguagem técnica apropriada
- Estruturação lógica de respostas clínicas

## Tecnologias Utilizadas

- **LangChain**: Framework para aplicações com LLM
- **Transformers**: Biblioteca da Hugging Face para modelos
- **Chroma**: Base de dados vetorial
- **PyTorch**: Framework de machine learning
- **BitsAndBytesConfig**: Quantização de modelos

## Modelos Utilizados

- **Embedding**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM**: `google/gemma-2-2b-it`

## Requisitos

### Dependências Python
```
torch
langchain
transformers
chromadb
huggingface_hub
python-dotenv
bitsandbytes
```

### Requisitos de Sistema
- Python 3.8+
- CUDA (recomendado para GPU)
- Mínimo 8GB RAM
- Token Hugging Face válido

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd vetllm
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
```bash
cp .env.example .env
```

4. Edite o arquivo `.env`:
```
HUGGINGFACE_TOKEN=seu_token_aqui
DIRETORIO_DB=caminho/para/base/dados/vetorial
```

## Uso

### Execução Básica
```bash
python vetllm.py
```

### Comandos Disponíveis
- **Pergunta normal**: Digite diretamente sua pergunta veterinária
- **Debug**: `debug [sua pergunta]` - Investiga o processo de busca
- **Sair**: `sair`, `exit` ou `quit`

### Exemplos de Perguntas
- "Quais são os sintomas de parvovirose canina?"
- "Como diagnosticar insuficiência renal em gatos?"
- "Protocolos de vacinação para filhotes"

## Funcionalidades

### 🔍 Busca Inteligente
O sistema utiliza a própria LLM para gerar termos de busca relevantes, incluindo:
- Termos técnicos em português e inglês
- Sinônimos médicos
- Abreviações comuns
- Variações de condições

### 🎯 Filtros de Relevância
- Filtragem automática de conteúdo irrelevante
- Detecção de padrões veterinários
- Validação de contexto com IA

### 📊 Sistema de Scoring
- Normalização de scores de similaridade
- Ordenação por relevância
- Controle de duplicatas

## Status do Projeto

### ✅ Funcionalidades Implementadas
- [x] Busca vetorial com embeddings
- [x] Geração de termos com LLM
- [x] Filtros de relevância
- [x] Sistema de debug
- [x] Interface de linha de comando

### 🔄 Em Desenvolvimento
- [ ] Interface web
- [ ] API REST
- [ ] Expansão da base de dados
- [ ] Métricas de avaliação
- [ ] Suporte a imagens médicas

### 🎯 Próximas Versões
- [ ] Sistema de feedback
- [ ] Integração com bases externas
- [ ] Modo offline completo
- [ ] Suporte multilíngue expandido

## Limitações Conhecidas

⚠️ **Importante**: Este sistema está em desenvolvimento e pode apresentar:
- Respostas imprecisas ocasionais
- Limitações na base de dados atual
- Possíveis erros de busca em casos específicos
- Dependência de conexão com Hugging Face

## Contribuição

Este é um projeto em desenvolvimento ativo. Contribuições são bem-vindas:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

O projeto está sob licensa [MIT](./LICENSE).

## Contato

* **Patrícia Canossa**
  * [Github:](https://github.com/pcanossa)
  * [LinkedIn:](https://www.linkedin.com/in/patricia-canossa-gagliardi/) 

## Aviso Legal

⚠️ **Este sistema é uma ferramenta de apoio e não substitui o julgamento clínico profissional. Sempre consulte literatura atualizada e diretrizes veterinárias oficiais para decisões clínicas.**

---

**Status**: 🚧 Em Produção e Testes  
**Última Atualização**: [Data]  
**Versão**: 0.1.0-beta