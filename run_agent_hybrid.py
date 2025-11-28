# run_agent_hybrid.py

import json
import os
from typing import Dict, Any

import click
import dspy

from agent.graph_hybrid import build_hybrid_graph




import dspy

def configure_lm() -> None:
    """
    Configure DSPy to use local Ollama model qwen2.5:3b.
    """
    lm = dspy.LM(
        "ollama_chat/qwen2.5:3b",      
        api_base="http://localhost:11434",  
        api_key="",                    
        max_tokens=2048,
        temperature=0.2,
    )
    dspy.configure(lm=lm)




# --------- Helpers --------- #


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------- CLI --------- #


@click.command()
@click.option("--batch", "batch_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_path", required=True, type=click.Path())
def main(batch_path: str, out_path: str):
    """
    Example:
    python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
    """

    configure_lm()


    project_root = os.path.dirname(os.path.abspath(__file__))

    db_path = os.path.join(project_root, "data", "northwind.sqlite")
    docs_dir = os.path.join(project_root, "docs")

    print("Using DB path:", db_path)
    print("Using docs dir:", docs_dir)

    app = build_hybrid_graph(db_path=db_path, docs_dir=docs_dir)

    outputs = []

    for item in load_jsonl(batch_path):
        qid = item["id"]
        question = item["question"]
        format_hint = item.get("format_hint", "")

        state: Dict[str, Any] = {
            "id": qid,
            "question": question,
            "format_hint": format_hint,
            "attempts": 0,
            "citations": [],
        }

        final_state = app.invoke(state)

        out_row = {
            "id": qid,
            "final_answer": final_state.get("final_answer"),
            "sql": final_state.get("sql", "") or "",
            "confidence": float(final_state.get("confidence", 0.0)),
            "explanation": final_state.get("explanation", "")[:300],
            "citations": final_state.get("citations", []),
        }
        outputs.append(out_row)

    write_jsonl(out_path, outputs)
    print(f"Wrote {len(outputs)} answers to {out_path}")


if __name__ == "__main__":
    main()
