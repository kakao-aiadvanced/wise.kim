import streamlit as st
from tavily import TavilyClient
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document

tavily = TavilyClient(api_key="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")

st.set_page_config(
    page_title="Research Assistant",
    page_icon=":orange_heart:",
)


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]


def main():
    # 모델 선택
    llm_model = st.sidebar.selectbox(
        "Select Model",
        options=[
            "llama3",
        ],
    )

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # 웹 페이지 로드 및 문서 분할
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # 벡터 저장소 생성
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
    )
    retriever = vectorstore.as_retriever()

    # RAG 에이전트 노드 및 엣지 정의
    def retrieve(state):
        print("---RETRIEVE---")
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"context": documents, "question": question})
        retry_count = state.get("retry_count", 0)
        return {"documents": documents, "question": question, "generation": generation, "retry_count": retry_count + 1}

    def grade_documents(state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
        }

    def web_search(state):
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]
        docs = tavily.search(query=question)["results"]
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}

    def route_question(state):
        print("---ROUTE QUESTION---")
        question = state["question"]
        print(question)
        source = question_router.invoke({"question": question})
        print(source)
        print(source["datasource"])
        if source["datasource"] == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source["datasource"] == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    def decide_to_generate(state):
        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(state):
        print("---CHECK HALLUCINATIONS---")
        retry_count = state.get("retry_count", 0)
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]


        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score["score"]

        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---", retry_count)
            if retry_count >= 1:
                print("❌ Too many retries. Halting.")
                return "failed: hallucination"
            return "not supported"

    # RAG 에이전트 그래프 구성
    workflow = StateGraph(GraphState)
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
            "failed: hallucination": END,
        },
    )

    app = workflow.compile()

    # rag_chain 정의
    llm = ChatOllama(model=llm_model, temperature=0)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question}
        Context: {context}
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )
    rag_chain = prompt | llm | StrOutputParser()

    # retrieval_grader, hallucination_grader, answer_grader 정의
    llm = ChatOllama(model=llm_model, format="json", temperature=0)

    retrieval_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )
    retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

    hallucination_grader_prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents}
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )
    hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

    answer_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation}
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )
    answer_grader = answer_grader_prompt | llm | JsonOutputParser()

    question_router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents,
    prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search'
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    question_router = question_router_prompt | llm | JsonOutputParser()

    # ----------------------------------------------------------------------
    # Streamlit 앱 UI
    st.title("Research Assistant powered by OpenAI")

    input_topic = st.text_input(
        ":female-scientist: Enter a topic",
        value="Superfast Llama 3 inference on Groq Cloud",
    )

    generate_report = st.button("Generate Report")

    if generate_report:
        with st.spinner("Generating Report"):
            inputs = {"question": input_topic}
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(f"Finished running: {key}:")
            # final_report = value["generation"]
            # st.markdown(final_report)
                    if key == "generate":
                        st.markdown("### ✅ Answer")
                        st.markdown(value["generation"])
                        # --- Show document sources ---
                        st.markdown("### 📄 출처")
                        documents = value.get("documents", [])
                        seen = set()
                        if documents:
                            for i, doc in enumerate(documents, 1):
                                url = doc.metadata.get("source", "").strip()
                                title = doc.metadata.get("title", "").strip() or url or "출처 없음"

                                # 중복 방지
                                key = (title, url)
                                if key in seen or not url:
                                    continue
                                seen.add(key)

                                st.markdown(f"{i}. [{title}]({url})")
                        else:
                            st.info("출처 정보가 없습니다.")

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.session_state.clear()
        st.experimental_rerun()


main()