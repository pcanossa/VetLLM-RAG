import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv
import os
import re

load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

# Constantes (mantém as mesmas do código original)
DIRETORIO_DB = os.getenv("DIRETORIO_DB")
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_LLM = "google/gemma-2-2b-it"

# Configuração (mantém a mesma)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quantize=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

# Carregar recursos (mantém o mesmo)
embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)
db = Chroma(persist_directory=DIRETORIO_DB, embedding_function=embeddings)
print("Banco de dados carregado com sucesso...")

print("🤖 Carregando modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_LLM)
model = AutoModelForCausalLM.from_pretrained(
    MODELO_LLM,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Modelo carregado com sucesso!")

def gerar_termos_busca_com_llm(pergunta):
    """Usa a LLM para gerar termos de busca inteligentes"""
    
    prompt_termos = f"""Você é um especialista em busca de documentos veterinários. 
Para a pergunta: "{pergunta}"

Gere 5-7 termos de busca específicos e relevantes, incluindo:
- Termos técnicos em português E inglês
- Sinônimos médicos
- Abreviações comuns
- Variações da condição

Formato: termo1, termo2, termo3, termo4, termo5

Pergunta: {pergunta}
Termos de busca:"""

    # Tokenizar e gerar
    inputs = tokenizer(prompt_termos, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extrair apenas os termos gerados
    termos_texto = resposta.split("Termos de busca:")[-1].strip()
    
    # Processar e limpar termos
    termos = []
    for termo in termos_texto.split(','):
        termo_limpo = termo.strip().lower()
        # Remover termos muito curtos ou irrelevantes
        if len(termo_limpo) > 2 and not termo_limpo.isdigit():
            termos.append(termo_limpo)
    
    # Fallback se a LLM não gerou termos adequados
    if not termos:
        palavras_pergunta = [p.strip() for p in pergunta.lower().split() if len(p.strip()) > 3]
        termos = palavras_pergunta[:5]
    
    print(f"🧠 LLM gerou os termos: {termos}")
    return termos[:7]  # Máximo 7 termos

def normalizar_score(score):
    """Normaliza scores negativos do Chroma"""
    if score < 0:
        return 1.0 / (1.0 + abs(score))
    return min(score, 1.0)

def filtrar_documento_relevante(doc):
    """Filtra documentos irrelevantes sem precisar listar manualmente"""
    conteudo = doc.page_content.strip()
    
    # Critérios automáticos de relevância
    if len(conteudo) < 50:  # Muito curto
        return False
    
    # Padrões que indicam conteúdo irrelevante
    padroes_irrelevantes = [
        r'^#\s*$',  # Apenas hashtag
        r'^\s*Foreword\s*$',  # Apenas "Foreword"
        r'^\s*References\s*$',  # Apenas "References"
        r'^\d+\.\s*\w+\s+\w+\.\s*\d{4}',  # Formato de referência bibliográfica
        r'^[A-Z][a-z]+\s+[A-Z]{2}.*Journal of',  # Padrão de citação
    ]
    
    for padrao in padroes_irrelevantes:
        if re.match(padrao, conteudo):
            return False
    
    # Se tem palavras relevantes para veterinária, provavelmente é bom
    palavras_veterinarias = [
        'dog', 'cat', 'canine', 'feline', 'veterinary', 'clinical', 'diagnosis', 
        'treatment', 'patient', 'cão', 'gato', 'canino', 'felino', 'veterinário', 
        'clínico', 'diagnóstico', 'tratamento', 'paciente', 'animal'
    ]
    
    conteudo_lower = conteudo.lower()
    palavras_encontradas = sum(1 for palavra in palavras_veterinarias if palavra in conteudo_lower)
    
    return palavras_encontradas >= 2  # Pelo menos 2 palavras veterinárias

def buscar_contexto_inteligente(pergunta, k=5):
    """Busca inteligente usando LLM para gerar termos"""
    print(f"🔍 Iniciando busca inteligente para: {pergunta}")
    
    # 1. Gerar termos de busca com a LLM
    termos_busca = gerar_termos_busca_com_llm(pergunta)
    
    # 2. Buscar com cada termo
    todos_docs = []
    scores_relevancia = []
    
    for termo in termos_busca:
        try:
            docs_termo = db.similarity_search_with_relevance_scores(termo, k=3)
            for doc, score in docs_termo:
                score_normalizado = normalizar_score(score)
                
                # Aplicar filtro de relevância
                if filtrar_documento_relevante(doc):
                    todos_docs.append(doc)
                    scores_relevancia.append(score_normalizado)
                    print(f"📄 '{termo}' -> Score: {score_normalizado:.3f} -> {doc.page_content[:100]}...")
                    
        except Exception as e:
            print(f"⚠️ Erro buscando '{termo}': {e}")
            # Fallback para busca simples
            try:
                docs_termo = db.similarity_search(termo, k=2)
                for doc in docs_termo:
                    if filtrar_documento_relevante(doc):
                        todos_docs.append(doc)
                        scores_relevancia.append(0.5)
            except:
                continue
    
    print(f"📊 Total documentos relevantes: {len(todos_docs)}")
    
    # 3. Se não encontrou nada, busca direta pela pergunta completa
    if not todos_docs:
        print("🔄 Busca direta pela pergunta...")
        try:
            docs_diretos = db.similarity_search(pergunta, k=5)
            for doc in docs_diretos:
                if filtrar_documento_relevante(doc):
                    todos_docs.append(doc)
                    scores_relevancia.append(0.3)
        except:
            return "NENHUM DOCUMENTO RELEVANTE ENCONTRADO"
    
    # 4. Remover duplicatas (baseado em hash do conteúdo)
    docs_unicos = {}
    for i, doc in enumerate(todos_docs):
        conteudo_hash = hash(doc.page_content[:200])  # Hash dos primeiros 200 chars
        if conteudo_hash not in docs_unicos or scores_relevancia[i] > docs_unicos[conteudo_hash][1]:
            docs_unicos[conteudo_hash] = (doc, scores_relevancia[i])
    
    # 5. Ordenar por relevância e pegar os melhores
    docs_ordenados = sorted(docs_unicos.values(), key=lambda x: x[1], reverse=True)
    docs_finais = [doc for doc, _ in docs_ordenados[:k]]
    
    if not docs_finais:
        return "NENHUM DOCUMENTO RELEVANTE ENCONTRADO"
    
    # 6. Construir contexto estruturado
    contexto_parts = []
    for i, doc in enumerate(docs_finais):
        fonte = doc.metadata.get('source', 'Fonte desconhecida')
        if '\\' in fonte:
            fonte = fonte.split('\\')[-1]
        
        pagina = doc.metadata.get('page', 'N/A')
        titulo = doc.metadata.get('title', 'N/A')
        
        contexto_parts.append(
            f"DOCUMENTO {i+1} (Fonte: {fonte}, Página: {pagina}):\n"
            f"{doc.page_content.strip()}\n"
        )
    
    contexto_final = "\n".join(contexto_parts)
    
    # 7. Controle de tamanho
    if len(contexto_final) > 4000:  # Limite maior para contextos ricos
        contexto_final = contexto_final[:4000] + "...[CONTEXTO TRUNCADO]"
    
    print(f"✅ Contexto construído: {len(docs_finais)} docs, {len(contexto_final)} chars")
    return contexto_final

def validar_contexto_inteligente(contexto, pergunta):
    """Usa a LLM para validar se o contexto é relevante"""
    
    if "NENHUM DOCUMENTO RELEVANTE ENCONTRADO" in contexto:
        return False
    
    # Validação rápida baseada em conteúdo
    if len(contexto.strip()) < 100:
        return False
    
    prompt_validacao = f"""Pergunta: {pergunta}
Contexto encontrado: {contexto[:500]}...

O contexto contém informações relevantes para responder a pergunta? Responda apenas: SIM ou NÃO

Resposta:"""

    inputs = tokenizer(prompt_validacao, return_tensors="pt", max_length=800, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resposta_final = resposta.split("Resposta:")[-1].strip().upper()
    
    # Se a LLM disse SIM, considera válido
    if "SIM" in resposta_final:
        return True
    
    # Validação de backup: se tem palavras da pergunta no contexto
    palavras_pergunta = [p for p in pergunta.lower().split() if len(p) > 3]
    contexto_lower = contexto.lower()
    intersecoes = sum(1 for palavra in palavras_pergunta if palavra in contexto_lower)
    
    return intersecoes >= 2

def gerar_resposta_final_inteligente(pergunta):
    """Pipeline completo de busca e geração inteligente"""
    print("🚀 Pipeline inteligente iniciado...")
    
    # 1. Buscar contexto com LLM
    contexto = buscar_contexto_inteligente(pergunta, k=4)
    
    # 2. Validar contexto com LLM
    if not validar_contexto_inteligente(contexto, pergunta):
        return (
            f"⚠️ Não encontrei informações suficientemente relevantes sobre '{pergunta}' "
            f"na base de dados atual.\n\n"
            f"💡 Sugestões:\n"
            f"• Tente reformular a pergunta com termos mais específicos\n"
            f"• Use o comando 'debug [sua pergunta]' para investigar a busca\n"
            f"• Verifique se há documentos sobre este tópico na base de dados"
        )
    
    print("🤖 Gerando resposta final...")
    
    # 3. Template (mantém o mesmo do código original)
    template_prompt = """
## Identidade e Personalidade
Você é um assistente de IA especialista em medicina veterinária. Sua tarefa é responder à pergunta a seguir de forma clara, estruturada e baseada em raciocínio clínico, usando exclusivamente o contexto fornecido.

## Contexto e Propósito
Você é especialista em todas as áreas veterinárias, desde cuidados preventivos até tratamentos complexos. Seu objetivo principal é entender as necessidades e dúvidas do veterinário para oferecer informações técnicas, precisas e relevantes.

Contexto: {contexto}

## Instruções Principais
- Responda APENAS com informações baseadas no contexto fornecido
- Não adicione informações externas ou opiniões pessoais
- Se a informação não estiver no contexto, declare isso claramente
- Use linguagem técnica apropriada mas clara
- Estruture em categorias lógicas quando apropriado

Pergunta: {pergunta}

**RESPOSTA:**"""

    # 4. Gerar resposta
    prompt_completo = template_prompt.format(contexto=contexto, pergunta=pergunta)
    
    # Tokenização (mantém a lógica original)
    if "DialoGPT" in MODELO_LLM:
        inputs = tokenizer(prompt_completo, return_tensors="pt", truncation=True, max_length=1400)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_usado = prompt_completo
    else:
        messages = [{"role": "user", "content": prompt_completo}]
        prompt_formatado = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_formatado, return_tensors="pt", truncation=True, max_length=1400)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_usado = prompt_formatado
    
    # Gerar resposta
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decodificar
    resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extrair resposta
    if "DialoGPT" in MODELO_LLM:
        resposta_final = resposta_completa[len(prompt_usado):].strip()
    else:
        if "<start_of_turn>model" in resposta_completa:
            resposta_final = resposta_completa.split("<start_of_turn>model")[-1].strip()
        else:
            resposta_final = resposta_completa[len(prompt_usado):].strip()
    
    # Limpeza
    resposta_final = resposta_final.replace("**RESPOSTA:**", "").strip()
    
    if len(resposta_final) < 20:
        return "⚠️ Não consegui gerar uma resposta adequada. Tente reformular a pergunta."
    
    return resposta_final

def debug_busca_inteligente(pergunta):
    """Debug da busca inteligente"""
    print(f"🔍 DEBUG INTELIGENTE: {pergunta}")
    
    # Mostrar termos gerados pela LLM
    termos = gerar_termos_busca_com_llm(pergunta)
    print(f"🧠 Termos gerados pela LLM: {termos}")
    
    # Testar cada termo
    for termo in termos[:3]:
        print(f"\n📋 Testando termo: '{termo}'")
        try:
            docs = db.similarity_search(termo, k=2)
            for i, doc in enumerate(docs):
                relevante = "✅" if filtrar_documento_relevante(doc) else "❌"
                print(f"{relevante} Doc {i+1}: {doc.page_content[:150]}...")
        except Exception as e:
            print(f"❌ Erro: {e}")

# Loop principal atualizado
if __name__ == "__main__":
    print("🩺 VetLLM INTELIGENTE inicializado!")
    print("💡 Comandos: 'debug [pergunta]' | 'sair' para encerrar")
    
    while True:
        pergunta_usuario = input("\n📝 Sua pergunta: ").strip()
        
        if pergunta_usuario.lower() in ['sair', 'exit', 'quit']:
            print("👋 Encerrando...")
            break
        
        if pergunta_usuario.startswith('debug '):
            pergunta_debug = pergunta_usuario[6:]
            debug_busca_inteligente(pergunta_debug)
            continue
        
        if not pergunta_usuario:
            print("⚠️ Digite uma pergunta válida.")
            continue
        
        try:
            resposta = gerar_resposta_final_inteligente(pergunta_usuario)
            print(f"\n🤖 **Resposta:**\n{resposta}")
        except Exception as e:
            print(f"❌ Erro: {e}")
            print("💡 Tente usar o comando 'debug' para investigar.")