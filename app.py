"""
PIK BIM Assistant — Streamlit RAG чат-бот для BIM.
Данные: ./data/*.txt. Эмбеддинги: E5. Хранилище: FAISS. LLM: GigaChat.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from langchain_community.vectorstores import FAISS
from langchain_community.chat_models.gigachat import GigaChat
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


class E5Embeddings(HuggingFaceEmbeddings):
    """E5 требует префиксы для асимметричного поиска."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([E5_PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(E5_QUERY_PREFIX + text)


def load_txt_documents(data_dir: str) -> list[Document]:
    docs = []
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return docs
    for f in sorted(data_path.glob("*.txt")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": f.name}))
        except Exception as e:
            st.warning(f"Не удалось прочитать {f.name}: {e}")
    return docs


def build_faiss_index(docs: list[Document], embeddings: E5Embeddings):
    # ~500 токенов ≈ 2000 символов, перекрытие 100 ≈ 400
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return None
    return FAISS.from_documents(chunks, embeddings)


def get_retriever_and_llm():
    if "faiss_index" in st.session_state and st.session_state.faiss_index is not None:
        return (
            st.session_state.faiss_index.as_retriever(search_kwargs={"k": 5}),
            st.session_state.llm,
        )

    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    with st.spinner("Загрузка документов и построение индекса..."):
        docs = load_txt_documents(str(data_dir))
        if not docs:
            st.error("В папке ./data/ не найдено .txt файлов.")
            return None, None

        embeddings = E5Embeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        faiss_index = build_faiss_index(docs, embeddings)
        if faiss_index is None:
            st.error("Не удалось построить индекс (нет чанков).")
            return None, None

        credentials = os.environ.get("GIGACHAT_CREDENTIALS")
        if not credentials:
            st.error("В .env не задана переменная GIGACHAT_CREDENTIALS.")
            return None, None

        llm = GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
        )
        st.session_state.faiss_index = faiss_index
        st.session_state.llm = llm

    return faiss_index.as_retriever(search_kwargs={"k": 5}), llm


SYSTEM_PROMPT = (
    "Отвечай только на основе предоставленного контекста. "
    "Если ответа нет в контексте — скажи, что в источниках не найдено. Не придумывай факты."
)


def answer_with_sources(question: str, retriever, llm) -> tuple[str, list[dict]]:
    chunks = retriever.invoke(question)
    if not chunks:
        return "Подходящих фрагментов не найдено. Уточните вопрос или добавьте документы в ./data/.", []

    context = "\n\n---\n\n".join(doc.page_content for doc in chunks)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Контекст:\n\n{context}\n\nВопрос: {question}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    answer = response.content if hasattr(response, "content") else str(response)

    sources = [
        {
            "file": doc.metadata.get("source", "—"),
            "quote": (doc.page_content[:500] + "...") if len(doc.page_content) > 500 else doc.page_content,
        }
        for doc in chunks
    ]
    return answer, sources


# --- UI ---

st.set_page_config(page_title="PIK BIM Assistant", layout="wide")
st.title("PIK BIM Assistant (прототип)")

with st.sidebar:
    st.markdown("### О проекте")
    st.markdown(
        "RAG чат-бот по BIM: ответы на основе статей из `./data/`. "
        "Эмбеддинги E5, FAISS, ответы — GigaChat."
    )
    st.markdown("---")
    if st.session_state.get("faiss_index"):
        st.success("Индекс загружен")
    else:
        st.info("Задайте вопрос — при первом запросе построится индекс.")

retriever, llm = get_retriever_and_llm()
if retriever is None or llm is None:
    st.stop()

question = st.chat_input("Задайте вопрос...")
if question:
    answer, sources = answer_with_sources(question, retriever, llm)
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(answer)
    st.markdown("#### Источники")
    for src in sources:
        with st.expander(f"📄 {src['file']}"):
            st.caption("Цитата:")
            st.text(src["quote"])
