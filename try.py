from pathlib import Path
import os

import json
import shutil
import os
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



# BASE_DIR = Path(__file__).resolve().parent

# TEA_CSV_PATH = BASE_DIR / "data" / "tea_metadata.csv"
# DOCUMENT_JSON_PATH = (
#     BASE_DIR / "output" / "external_documents.jsonl"
# )

# print("CSV 경로:", TEA_CSV_PATH)
# print("JSON 경로:", DOCUMENT_JSON_PATH)
# print("CSV 존재:", TEA_CSV_PATH.is_file())
# print("JSON 존재:", DOCUMENT_JSON_PATH.is_file())


def load_tea_metadata() -> dict[str, dict]:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV 파일 없음: {CSV_PATH}")

    encodings = ["utf-8-sig", "cp949", "euc-kr"]

    for encoding in encodings:
        try:
            df = pd.read_csv(
                CSV_PATH,
                encoding=encoding,
                dtype={"tea_id": str},
            )

            print(f"\n[인코딩 확인] {encoding}")
            print(df[["tea_id", "name", "english_name"]].head(5).to_string(index=False))

        except (UnicodeDecodeError, LookupError) as error:
            print(f"{encoding} 읽기 실패: {error}")

    raise SystemExit(
        "인코딩 확인 완료. 출력 결과에서 한글이 정상인 인코딩을 확인하세요."
    )