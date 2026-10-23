import json
import shutil
import unicodedata
from pathlib import Path

import pandas as pd
import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"설정 파일이 없습니다: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml의 최상위 구조는 YAML 객체여야 합니다.")

    return config


CONFIG = load_config()

CSV_PATH = BASE_DIR / CONFIG["paths"]["csv"]
JSONL_PATH = BASE_DIR / CONFIG["paths"]["jsonl"]
DB_PATH = BASE_DIR / CONFIG["paths"]["vectorstore"]

COLLECTION_NAME = CONFIG["vectorstore"]["collection_name"]


def normalize_name(value) -> str:
    """차 이름 비교용 정규화."""
    if value is None or pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    return "".join(value.casefold().split())


def scalar_value(value):
    """Chroma 메타데이터에 저장 가능한 값으로 변환."""
    if value is None or pd.isna(value):
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def load_tea_metadata() -> dict[str, dict]:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV 파일 없음: {CSV_PATH}")

    df = pd.read_csv(
        CSV_PATH,
        encoding="utf-8-sig",
        dtype=str,
    )

    df.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in df.columns
    ]

    if "name" not in df.columns:
        raise ValueError("CSV에 name 열이 필요합니다.")

    df = df.fillna("")
    df["name"] = df["name"].map(
        lambda value: unicodedata.normalize(
            "NFKC", str(value)
        ).strip()
    )

    df = df[df["name"] != ""]
    df = df.drop_duplicates(subset=["name"], keep="first")

    metadata = {}

    for record in df.to_dict(orient="records"):
        key = normalize_name(record.get("name"))

        if key:
            metadata[key] = record

    print(f"CSV 차 메타데이터: {len(metadata)}개")
    return metadata


def load_external_documents() -> list[dict]:
    if not JSONL_PATH.is_file():
        raise FileNotFoundError(
            f"외부 문서 파일 없음: {JSONL_PATH}"
        )

    raw_text = JSONL_PATH.read_text(encoding="utf-8-sig")

    if not raw_text.strip():
        raise ValueError("외부 문서 파일이 비어 있습니다.")

    # JSONL을 우선 읽고, 일반 JSON 객체/배열도 지원한다.
    try:
        records = [
            json.loads(line)
            for line in raw_text.splitlines()
            if line.strip()
        ]

        if not all(isinstance(item, dict) for item in records):
            raise ValueError("JSONL 각 줄은 JSON 객체여야 합니다.")

    except json.JSONDecodeError:
        raw = json.loads(raw_text)

        if isinstance(raw, dict):
            records = [raw]
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError("지원하지 않는 JSON 구조입니다.")

    records = [
        record
        for record in records
        if isinstance(record, dict)
        and str(record.get("content") or "").strip()
    ]

    if not records:
        raise ValueError("content가 있는 외부 문서가 없습니다.")

    print(f"외부 문서: {len(records)}개")
    return records


def join_documents(
    records: list[dict],
    tea_metadata: dict[str, dict],
) -> list[Document]:
    documents = []
    matched = 0
    unmatched = 0

    for index, record in enumerate(records):
        tea_name_from_record = (
            record.get("tea_name")
            or record.get("tname")
            or record.get("name")
            or ""
        )

        tea = tea_metadata.get(
            normalize_name(tea_name_from_record),
            {},
        )

        if tea:
            matched += 1
        else:
            unmatched += 1

        tea_name = (
            str(tea.get("name") or "").strip()
            or str(tea_name_from_record or "").strip()
        )

        title = (
            record.get("document_title")
            or record.get("title")
            or tea_name
        )

        # 원본 문서 식별자. 청크 ID는 split_documents에서 별도 생성한다.
        document_id = str(
            record.get("document_id")
            or f"doc-{index}"
        ).strip()

        source_url = str(
            record.get("source_url")
            or record.get("url")
            or ""
        ).strip()

        source_type = str(
            record.get("source_type")
            or "external"
        ).strip()

        context_fields = [
            ("차 이름", tea_name),
            ("영문 이름", tea.get("english_name")),
            ("카테고리", tea.get("category")),
            ("하위 카테고리", tea.get("subcategory")),
            ("원재료", tea.get("ingredients")),
            ("설명", tea.get("description")),
            ("카페인 수준", tea.get("caffeine_level")),
            (
                "카페인 함량(mg)",
                tea.get("caffeine_mg_per_tea_amount"),
            ),
            ("맛", tea.get("taste")),
            ("향", tea.get("aroma")),
            ("우리기 방법", tea.get("brewing_method")),
            ("물 온도(°C)", tea.get("water_temperature_c")),
            ("우리는 시간(분)", tea.get("steep_time_min")),
            ("외부 문서 제목", title),
        ]

        context = "\n".join(
            f"{label}: {value}"
            for label, value in context_fields
            if value is not None
            and not pd.isna(value)
            and str(value).strip()
        )

        content = str(record["content"]).strip()
        page_content = (
            f"{context}\n\n외부 문서 본문:\n{content}"
        )

        metadata = {
            "document_id": document_id,
            "tea_name": tea_name,
            "document_title": str(title),
            "source_url": source_url,
            "source_type": source_type,
            "chunk_source_index": index,
            "category": str(tea.get("category")or ""),
        }

        for key in (
            "category",
            "subcategory",
            "caffeine_level",
            "taste",
            "aroma",
            "brewing_method",
        ):
            value = tea.get(key)

            if value is not None and not pd.isna(value):
                metadata[key] = str(value)

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    print(f"CSV 연결 성공: {matched}개")
    print(f"CSV 연결 실패: {unmatched}개")

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    settings = CONFIG["chunking"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(settings["chunk_size"]),
        chunk_overlap=int(settings["chunk_overlap"]),
        separators=settings["separators"],
        add_start_index=True,
    )

    chunks = []
    chunk_counts = {}

    for document in documents:
        source_chunks = splitter.split_documents([document])

        for chunk_index, chunk in enumerate(source_chunks):
            document_id = chunk.metadata["document_id"]
            source_index = chunk.metadata["chunk_source_index"]

            # 원문 ID가 중복되어도 각 입력 레코드를 구별한다.
            chunk_id = (
                f"{document_id}-{source_index}-{chunk_index}"
            )

            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = chunk_id

            chunks.append(chunk)

        chunk_counts[str(document.metadata["document_id"])] = (
            chunk_counts.get(
                str(document.metadata["document_id"]), 0
            ) + len(source_chunks)
        )

    print(f"분할 후 청크 수: {len(chunks)}개")
    return chunks


def build_index(chunks: list[Document]) -> Chroma:
    if not chunks:
        raise ValueError("인덱싱할 청크가 없습니다.")

    embedding_config = CONFIG["embedding"]

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_config["model_name"],
        model_kwargs={
            "device": embedding_config.get("device", "cpu")
        },
        encode_kwargs={
            "normalize_embeddings": embedding_config.get(
                "normalize_embeddings", True
            )
        },
    )

    batch_size = int(
        CONFIG["vectorstore"].get("batch_size", 32)
    )

    if batch_size <= 0:
        raise ValueError(
            "vectorstore.batch_size는 1 이상이어야 합니다."
        )

    # Chroma ID 중복으로 인한 덮어쓰기를 방지한다.
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "중복 chunk_id가 발견되었습니다. "
            "문서 ID와 청크 ID 생성 규칙을 확인하세요."
        )

    # 전용 인덱스 폴더만 삭제한다.
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    vectorstore = None

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        batch_ids = [
            chunk.metadata["chunk_id"]
            for chunk in batch
        ]

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                ids=batch_ids,
                collection_name=COLLECTION_NAME,
                persist_directory=str(DB_PATH),
            )
        else:
            vectorstore.add_documents(
                documents=batch,
                ids=batch_ids,
            )

        print(
            f"인덱싱 진행: "
            f"{min(start + batch_size, len(chunks))}/{len(chunks)}"
        )

    print(f"컬렉션: {COLLECTION_NAME}")
    print(f"인덱스 저장 완료: {DB_PATH}")
    return vectorstore


def test_search(vectorstore: Chroma):
    query = "센차의 맛과 카페인, 우리는 방법을 알려 줘"

    results = vectorstore.similarity_search(query, k=3)

    print(f"\n테스트 질문: {query}")

    for index, document in enumerate(results, start=1):
        print(f"\n--- 검색 결과 {index} ---")
        print("차:", document.metadata.get("tea_name"))
        print("문서:", document.metadata.get("document_title"))
        print("출처:", document.metadata.get("source_url"))
        print("출처 종류:", document.metadata.get("source_type"))
        print("문서 ID:", document.metadata.get("document_id"))
        print("청크 ID:", document.metadata.get("chunk_id"))
        print("본문:", document.page_content[:350])


if __name__ == "__main__":
    print("=== Tea RAG 인덱스 구축 시작 ===")

    tea_metadata = load_tea_metadata()
    records = load_external_documents()
    documents = join_documents(records, tea_metadata)
    chunks = split_documents(documents)
    vectorstore = build_index(chunks)

    test_search(vectorstore)

    print("\n=== 인덱스 구축 완료 ===")