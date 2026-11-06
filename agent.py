"""LLM-driven agent: python agent.py "Verify all letters in samples against samples/master_signature.pdf" """
import sys

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

import config
import pipeline

_STATE: dict = {"store": {}, "rows": []}

SYSTEM_PROMPT = """You verify customer letter signatures against a bank master signature register.
Steps: 1) call load_master once, 2) call list_letters, 3) call verify_letter for every letter,
4) call save_results, 5) summarise MATCH / MISMATCH / REVIEW / NOT_FOUND counts and list the non-MATCH CIFs.
Never invent scores; report only tool output."""


@tool
def load_master(master_pdf: str) -> str:
    """Extract every CIF and its master signature from the master signature PDF (text or scanned)."""
    _STATE["store"] = pipeline.build_master_store(master_pdf)
    _STATE["rows"] = []
    return f"Loaded {len(_STATE['store'])} master signatures: {', '.join(sorted(_STATE['store']))}"


@tool
def list_letters(letters_dir: str) -> str:
    """List letter PDF files in a folder (master PDFs excluded)."""
    from pathlib import Path
    files = sorted(p for p in Path(letters_dir).glob("*.pdf") if "master" not in p.name.lower())
    return "\n".join(str(p) for p in files) or "No letters found."


@tool
def verify_letter(letter_pdf: str) -> str:
    """Extract the CIF and signature from one letter PDF and compare it with the master signature."""
    row = pipeline.verify_letter(letter_pdf, _STATE["store"])
    _STATE["rows"].append(row)
    return str(row)


@tool
def save_results() -> str:
    """Write all verification results to output/cif_list.csv."""
    out = pipeline.write_csv(_STATE["rows"], config.OUTPUT_DIR / "cif_list.csv")
    return f"Saved {len(_STATE['rows'])} rows to {out}"


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        f"Verify all letters in {config.SAMPLES_DIR} against {config.SAMPLES_DIR / 'master_signature.pdf'}")
    llm = ChatOllama(model=config.OLLAMA_AGENT_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
    agent = create_agent(model=llm, tools=[load_master, list_letters, verify_letter, save_results], system_prompt=SYSTEM_PROMPT)
    result = agent.invoke({"messages": [("user", prompt)]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
