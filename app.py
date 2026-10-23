
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="Tea Lab - Tea RAG",
    page_icon="🍵",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

load_dotenv(BASE_DIR / ".env")


def load_config() -> dict:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"설정 파일이 없습니다: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml의 최상위 구조는 YAML 객체여야 합니다."
        )

    return config


CONFIG = load_config()

CSV_PATH = BASE_DIR / CONFIG["paths"]["csv"]
VECTOR_DIR = BASE_DIR / CONFIG["paths"]["vectorstore"]

COLLECTION_NAME = CONFIG["vectorstore"]["collection_name"]

EMBEDDING_CONFIG = CONFIG["embedding"]
LLM_CONFIG = CONFIG["llm"]
RETRIEVAL_CONFIG = CONFIG.get("retrieval", {})

METADATA_LIMIT = int(
    RETRIEVAL_CONFIG.get("metadata_limit", 5)
)
EXTERNAL_LIMIT = int(
    RETRIEVAL_CONFIG.get("external_limit", 5)
)

if METADATA_LIMIT <= 0 or EXTERNAL_LIMIT <= 0:
    raise ValueError(
        "retrieval의 metadata_limit와 external_limit은 "
        "1 이상이어야 합니다."
    )


COLUMNS = [
    "name",
    "english_name",
    "category",
    "subcategory",
    "ingredients",
    "description",
    "caffeine_level",
    "caffeine_mg_per_tea_amount",
    "tea_amount_g",
    "water_volume_ml",
    "water_temperature_c",
    "steep_time_min",
    "brewing_method",
    "taste",
    "aroma",
    "source_url",
]

CITE_PATTERN = re.compile(r"\[근거\s*([\d,\s]+)\]")


def normalize_name(value) -> str:
    """차 이름 비교를 위한 유니코드 정규화."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    value = unicodedata.normalize("NFKC", str(value))
    return "".join(value.casefold().split())

KOREAN_PARTICLES = (
    "으로부터", "에게서", "한테서", "에서부터",
    "으로써", "으로서", "이라도", "이어서",
    "에게", "한테", "께서", "부터", "까지",
    "처럼", "보다", "마다", "조차", "마저",
    "밖에", "이나", "라도", "든지", "든가",
    "은", "는", "이", "가", "을", "를",
    "의", "에", "와", "과", "도", "로", "만",
    "랑", "께", "야", "요",
)


def normalize_query_tokens(question: str) -> list[str]:
    """질문에서 한국어 조사를 제거한 검색 토큰을 반환한다."""

    text = unicodedata.normalize("NFKC", question).casefold()

    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)
    normalized_tokens = []

    for token in tokens:
        # 긴 조사부터 확인하여 짧은 조사와의 충돌을 줄인다.
        for particle in sorted(
            KOREAN_PARTICLES,
            key=len,
            reverse=True,
        ):
            if (
                token.endswith(particle)
                and len(token) > len(particle)
            ):
                token = token[:-len(particle)]
                break

        if token:
            normalized_tokens.append(token)

    return normalized_tokens


# --------------------------------------------------
# 2. CSV 로딩
# --------------------------------------------------

@st.cache_data
def load_teas(path: str) -> list[dict]:
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"CSV 파일이 없습니다: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        dtype=str,
    )

    df.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in df.columns
    ]

    missing = [
        column for column in COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"CSV 필수 열이 없습니다: {missing}"
        )

    df = df.fillna("")

    df["name"] = df["name"].astype(str).map(
        lambda value: unicodedata.normalize(
            "NFKC", value
        ).strip()
    )

    df["english_name"] = (
        df["english_name"].astype(str).str.strip()
    )

    df = df[df["name"] != ""]

    df["_normalized_name"] = df["name"].map(normalize_name)
    df = df.drop_duplicates(
        subset=["_normalized_name"],
        keep="first",
    )
    df = df.drop(columns=["_normalized_name"])

    return df.to_dict(orient="records")


def tea_to_text(tea: dict) -> str:
    """CSV 한 행을 검색 및 LLM 입력용 텍스트로 변환한다."""
    return "\n".join(
        f"{column}: {tea.get(column, '')}"
        for column in COLUMNS
        if str(tea.get(column, "")).strip()
    )


# --------------------------------------------------
# 3. LLM 및 벡터 인덱스
# --------------------------------------------------

@st.cache_resource
def get_model():
    key = os.getenv("GMS_KEY")

    if not key:
        raise ValueError(
            "GMS_KEY가 없습니다. 프로젝트 루트의 .env를 확인하세요."
        )

    return ChatOpenAI(
        model=LLM_CONFIG.get("model", "gpt-5.4-nano"),
        api_key=key,
        base_url=LLM_CONFIG.get(
            "base_url",
            "https://gms.ssafy.io/gmsapi/api.openai.com/v1/",
        ),
        temperature=float(
            LLM_CONFIG.get("temperature", 0)
        ),
    )


@st.cache_resource
def get_vectorstore():
    if not VECTOR_DIR.is_dir():
        raise FileNotFoundError(
            f"Chroma 인덱스가 없습니다: {VECTOR_DIR}\n"
            "먼저 python build_index.py를 실행하세요."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_CONFIG["model_name"],
        model_kwargs={
            "device": EMBEDDING_CONFIG.get("device", "cpu")
        },
        encode_kwargs={
            "normalize_embeddings": EMBEDDING_CONFIG.get(
                "normalize_embeddings", True
            )
        },
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )


# --------------------------------------------------
# 4. 전체 차 메타데이터 검색
# --------------------------------------------------

def search_tea_metadata(
    question: str,
    teas: list[dict],
    limit: int = METADATA_LIMIT,
) -> list[dict]:
    """차 이름, 영문명, 카테고리, 설명 등을 기준으로 검색한다."""

    query = normalize_name(question)

    # tokens = set(
    #     re.findall(
    #         r"[가-힣a-zA-Z0-9]+",
    #         question.casefold(),
    #     )
    # )
    tokens = set(normalize_query_tokens(question))

    scored = []

    for tea in teas:
        name = normalize_name(tea.get("name", ""))
        english_name = normalize_name(
            tea.get("english_name", "")
        )

        fields = [
            tea.get("name", ""),
            tea.get("english_name", ""),
            tea.get("category", ""),
            tea.get("subcategory", ""),
            tea.get("ingredients", ""),
            tea.get("description", ""),
            tea.get("caffeine_level", ""),
            tea.get("caffeine_mg_per_tea_amount", ""),
            tea.get("tea_amount_g", ""),
            tea.get("taste", ""),
            tea.get("aroma", ""),
            tea.get("brewing_method", ""),
        ]

        searchable = " ".join(
            str(value) for value in fields if value
        ).casefold()

        score = 0

        if name and name in query:
            score += 20

        if english_name and english_name in query:
            score += 20

        for token in tokens:
            if len(token) >= 2 and token in searchable:
                score += 1

        if score > 0:
            scored.append((score, tea))

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        tea for _, tea in scored[:limit]
    ]

def search_by_category(
    question: str,
    teas: list[dict],
) -> tuple[str | None, list[dict]]:
    """질문에 명시된 카테고리를 CSV 전체에서 검색한다."""

    normalized_question = normalize_name(question)

    categories = {
        normalize_name(tea.get("category", "")):
        str(tea.get("category", "")).strip()
        for tea in teas
        if str(tea.get("category", "")).strip()
    }

    # 질문에 실제로 등장하는 카테고리를 확인한다.
    matched_categories = [
        original
        for normalized, original in categories.items()
        if normalized and normalized in normalized_question
    ]

    if not matched_categories:
        return None, []

    # 긴 카테고리명을 우선하여 부분 일치 충돌을 줄인다.
    category = max(matched_categories, key=len)

    results = [
        tea for tea in teas
        if normalize_name(tea.get("category", "")) ==
        normalize_name(category)
    ]

    return category, results

# --------------------------------------------------
# 5. Chroma 외부 문서 검색
# --------------------------------------------------

def search_external_documents(
    question: str,
    k: int = EXTERNAL_LIMIT,
    tea_name: str | None = None,
) -> list[dict]:
    """Chroma에서 외부 문서를 검색하고 출처 메타데이터를 전달한다."""

    vectorstore = get_vectorstore()

    search_kwargs = {"k": k}

    if tea_name and tea_name.strip():
        search_kwargs["filter"] = {
            "tea_name": tea_name.strip()
        }

    docs = vectorstore.similarity_search(
        question,
        **search_kwargs,
    )

    results = []

    for doc in docs:
        metadata = doc.metadata or {}

        document_tea_name = (
            metadata.get("tea_name")
            or metadata.get("name")
            or ""
        )

        results.append({
            "title": (
                metadata.get("document_title")
                or metadata.get("title")
                or document_tea_name
                or "외부 문서"
            ),
            "url": (
                metadata.get("source_url")
                or metadata.get("url")
                or ""
            ),
            "text": doc.page_content,
            "name": str(document_tea_name).strip(),
            "kind": "external",
            "source_type": metadata.get(
                "source_type", "external"
            ),
            "document_id": str(
                metadata.get("document_id") or ""
            ),
            "chunk_id": str(
                metadata.get("chunk_id") or ""
            ),
            "chunk_index": metadata.get(
                "chunk_index", -1
            ),
        })

    return results


# --------------------------------------------------
# 6. 근거 번호 및 출처 표시
# --------------------------------------------------

def cited_sources(
    answer: str,
    sources: list[dict],
) -> list[tuple[int, dict]]:
    """답변에서 인용한 번호만 원래 번호대로 반환한다."""

    cited_numbers = {
        int(number)
        for group in CITE_PATTERN.findall(answer)
        for number in group.split(",")
        if number.strip().isdigit()
    }

    return [
        (number, source)
        for number, source in enumerate(sources, start=1)
        if number in cited_numbers
    ]


def render_sources(
    answer: str,
    sources: list[dict],
) -> None:
    used = cited_sources(answer, sources)

    if not used:
        st.caption(
            "ⓘ 답변에 인용된 근거가 없습니다."
        )
        return

    groups = {}

    for number, source in used:
        document_id = source.get("document_id", "")

        if document_id:
            group_key = (
                f"doc:{document_id}"
                if source.get("kind") == "external"
                else f"metadata:{document_id}"
            )
        else:
            group_key = f"source:{number}"

        if group_key not in groups:
            groups[group_key] = {
                "numbers": [],
                "source": source,
                "texts": [],
                "urls": [],
            }

        group = groups[group_key]
        group["numbers"].append(number)

        text = source.get("text", "")
        if text and text not in group["texts"]:
            group["texts"].append(text)

        url = source.get("url", "")
        if (
            url.startswith(("http://", "https://"))
            and url not in group["urls"]
        ):
            group["urls"].append(url)

    with st.expander(
        f"답변에 사용한 근거 {len(used)}건"
    ):
        for group in groups.values():
            source = group["source"]

            numbers = ", ".join(
                str(number)
                for number in group["numbers"]
            )

            st.markdown(
                f"**[근거 {numbers}] {source['title']}**"
            )

            source_type = source.get(
                "source_type",
                source.get("kind", "unknown"),
            )

            type_labels = {
                "external": "외부 문서",
                "csv_metadata": "CSV 메타데이터",
                "metadata": "CSV 메타데이터",
            }

            st.caption(
                "출처 종류: "
                f"{type_labels.get(source_type, source_type)}"
            )

            for url in group["urls"]:
                st.markdown(f"[원문 보기]({url})")

            combined_text = "\n\n".join(group["texts"])

            if combined_text:
                st.caption(combined_text[:2000])

# --------------------------------------------------
# 7. CSV와 외부 문서 연결
# --------------------------------------------------

def connect_documents_to_teas(
    tea_results: list[dict],
    external_results: list[dict],
) -> list[dict]:
    """
    차 이름을 기준으로 CSV와 외부 문서를 연결한다.

    이름이 일치하지 않는 문서는 특정 차에
    연결된 것으로 표시하지 않는다.
    """

    tea_names = {
        normalize_name(tea.get("name", "")): tea
        for tea in tea_results
        if normalize_name(tea.get("name", ""))
    }

    connected = []

    for doc in external_results:
        doc = dict(doc)

        document_name = normalize_name(
            doc.get("name", "")
        )

        if document_name and document_name in tea_names:
            doc["matched_tea_name"] = tea_names[
                document_name
            ].get("name", "")
        else:
            doc["matched_tea_name"] = ""

        connected.append(doc)

    return connected


# --------------------------------------------------
# 8. 근거 병합 및 RAG 답변 생성
# --------------------------------------------------

def answer_question(
    question: str,
    teas: list[dict],
    selected_tea: dict | None = None,
) -> tuple[str, list[dict]]:
    """
    CSV 메타데이터와 Chroma 검색 결과를 병합한 뒤
    LLM을 이용해 근거 기반 답변을 생성한다.
    """

    # 1. CSV 메타데이터 검색
    if selected_tea:
        # 선택한 차를 검색 순위와 관계없이 근거에 포함한다.
        tea_results = [selected_tea]
    else:
        tea_results = search_tea_metadata(
            question,
            teas,
            limit=METADATA_LIMIT,
        )

    # 2. Chroma 외부 문서 검색
    external_results = search_external_documents(
        question,
        k=EXTERNAL_LIMIT,
        tea_name=(
            selected_tea.get("name")
            if selected_tea
            else None
        ),
    )

    # 3. 검색 문서와 CSV의 차 이름 연결 정보 확인
    external_results = connect_documents_to_teas(
        tea_results,
        external_results,
    )

    # 4. CSV 메타데이터를 근거 목록에 추가
    sources = []

    for tea in tea_results:
        tea_name = tea.get("name", "이름 없음")

        sources.append({
            "title": (
                f"{tea_name} "
                f"({tea.get('english_name', '')}) 메타데이터"
            ),
            "url": tea.get("source_url", ""),
            "text": tea_to_text(tea),
            "kind": "metadata",
            "source_type": "csv_metadata",
            "name": tea_name,
            "document_id": (
                f"csv:{normalize_name(tea_name)}"
            ),
            "chunk_id": "",
        })

    # 5. 중복 외부 청크 제거
    seen_chunk_ids = set()
    seen_fallback_keys = set()

    for doc in external_results:
        chunk_id = doc.get("chunk_id", "")

        if chunk_id:
            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)

        else:
            # chunk_id가 없는 구형 색인을 위한 보조 중복 검사
            fallback_key = (
                doc.get("document_id", ""),
                doc.get("name", ""),
                doc.get("title", ""),
                doc.get("text", ""),
            )

            if fallback_key in seen_fallback_keys:
                continue

            seen_fallback_keys.add(fallback_key)

        sources.append(doc)

    if not sources:
        return (
            "질문과 관련된 차 데이터를 찾지 못했습니다. "
            "차 이름이나 질문을 조금 더 구체적으로 입력해 주세요.",
            [],
        )

    # 6. 근거 번호를 확정한 뒤 LLM에 전달
    context_parts = []

    for number, source in enumerate(sources, start=1):
        context_parts.append(
            f"[근거 {number}]\n"
            f"제목: {source.get('title', '제목 없음')}\n"
            f"출처 종류: "
            f"{source.get('source_type', source.get('kind', 'unknown'))}\n"
            f"차 이름: {source.get('name', '')}\n"
            f"출처 URL: {source.get('url', '')}\n"
            f"내용:\n{source.get('text', '')[:4000]}"
        )

    context = "\n\n".join(context_parts)

    # 7. 근거에 기반한 답변 생성
    prompt = f"""
너는 차(Tea) 전문 큐레이터다.
반드시 제공된 근거에 기반해 한국어로 답변한다.

규칙:
1. 근거에 없는 사실이나 수치를 만들어 내지 않는다.
2. CSV 메타데이터와 외부 문서의 정보를 구분한다.
3. CSV 메타데이터는 차의 속성 및 추출 조건을 확인하는 근거로 사용한다.
4. 외부 문서는 해당 문서에 실제로 기록된 내용의 근거로 사용한다.
5. 질문한 차와 근거의 차 이름이 다르면 정보를 섞지 않는다.
6. 선택한 차가 있다면 해당 차에 대한 근거를 우선 사용한다.
7. 관련 근거가 없으면 확인할 수 없다고 설명한다.
8. 답변에 사용한 근거에만 [근거 N]을 표시한다.
9. 근거 번호는 제공된 [근거 N] 번호를 그대로 사용한다.
10. 사용하지 않은 근거 번호를 임의로 붙이지 않는다.
11. 출처 종류와 문서 제목을 고려하여 근거를 설명한다.
12. tea_id는 사용하지 않고 차 이름을 기준으로 판단한다.

[검색된 근거]
{context}

[사용자 질문]
{question}
"""

    response = get_model().invoke(prompt)
    content = response.content

    if isinstance(content, str):
        answer = content

    elif isinstance(content, list):
        text_parts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
        ]
        answer = "\n".join(text_parts) or str(content)

    else:
        answer = str(content)

    return answer, sources


# --------------------------------------------------
# 9. Streamlit 화면
# --------------------------------------------------

st.title("🍵 Tea Lab")
st.caption(
    "차의 종류, 맛, 향, 카페인과 추출 방법을 "
    "데이터에 근거해 안내하는 서비스"
)

try:
    teas = load_teas(str(CSV_PATH))

except Exception as error:
    st.error(
        f"차 데이터를 불러오지 못했습니다: {error}"
    )
    st.stop()


st.success(
    f"차 메타데이터 {len(teas)}개를 불러왔습니다."
)


def tea_label(index: int) -> str:
    tea = teas[index]

    name = tea.get("name") or "이름 없음"
    english_name = tea.get("english_name") or ""

    if english_name:
        return f"{name} ({english_name})"

    return name


# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 10. 사이드바 및 차 선택
# --------------------------------------------------

with st.sidebar:
    st.header("🍃 차 데이터")

    st.caption(
        "특정 차를 선택하지 않아도 전체 차에 질문할 수 있습니다."
    )

    selected_index = st.selectbox(
        "차 정보 미리보기 (선택 사항)",
        options=[None] + list(range(len(teas))),
        format_func=lambda index: (
            "전체 차 검색"
            if index is None
            else tea_label(index)
        ),
        key="tea_preview",
    )

    if st.button(
        "대화 지우기",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


selected_tea = (
    teas[selected_index]
    if selected_index is not None
    else None
)


if selected_tea is None:
    st.subheader("전체 차 데이터 검색")

else:
    st.subheader(
        str(selected_tea.get("name", "이름 없음"))
    )

    if selected_tea.get("english_name"):
        st.caption(selected_tea["english_name"])

    with st.expander("선택한 차의 메타데이터 확인"):
        st.json(selected_tea)


# --------------------------------------------------
# 11. 이전 대화 및 인용 근거 표시
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(
                message["content"],
                message.get("sources", []),
            )


# --------------------------------------------------
# 12. 질문 처리
# --------------------------------------------------

question = st.chat_input(
    "예: 군산은침이 뭐야? 또는 카페인이 적은 차를 추천해 줘."
)


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.status(
                "차 데이터와 외부 문서를 검색합니다...",
                expanded=True,
            ) as status:

                st.write("1. CSV 메타데이터 검색")
                st.write("2. 선택한 차에 맞는 Chroma 문서 검색")
                st.write("3. 중복 청크 제거 및 근거 병합")
                st.write("4. 근거 번호를 포함한 답변 생성")

                answer, sources = answer_question(
                    question,
                    teas,
                    selected_tea=selected_tea,
                )

                status.update(
                    label=(
                        f"답변 생성 완료 · "
                        f"근거 후보 {len(sources)}건"
                    ),
                    state="complete",
                    expanded=False,
                )

            st.markdown(answer)

            render_sources(
                answer,
                sources,
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            st.error("답변 생성에 실패했습니다.")

            st.caption(
                f"오류 유형: {type(error).__name__}"
            )

            st.exception(error)
