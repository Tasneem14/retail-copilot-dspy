import json

import dspy
from dspy.teleprompt import BootstrapFewShot

from agent.dspy_signatures import NL2SQL
from agent.graph_hybrid import NL2SQLModule  


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_examples(path):
    data = list(load_jsonl(path))
    examples = []
    for e in data:
        ex = dspy.Example(
            question=e["question"],
            schema=e.get("schema", ""),
            constraints_json=e.get("constraints_json", "{}"),
            sql=e["sql"],
        ).with_inputs("question", "schema", "constraints_json")
        examples.append(ex)
    return examples


def sql_exact_match(example, pred, trace=None) -> float:
    """Metric: returns 1 if the SQL matches (after strip/lower), otherwise 0."""

    gold = example.sql.strip().lower()
    got = (getattr(pred, "sql", "") or "").strip().lower()
    return 1.0 if gold == got else 0.0


def accuracy_on(dataset, program) -> float:
    correct = 0
    for ex in dataset:
        pred = program(
            question=ex.question,
            schema=ex.schema,
            constraints_json=ex.constraints_json,
        )
        
        class P:
            pass
        p = P()
        p.sql = pred
        if sql_exact_match(ex, p) == 1.0:
            correct += 1
    return correct / len(dataset)


def main():
    
    dspy.configure(
        lm = dspy.LM(
            "ollama_chat/qwen2.5:3b",      
            api_base="http://localhost:11434",  
            api_key="",                   
            max_tokens=2048,
            temperature=0.2,
        )
    )

    train_examples = make_examples("agent/training/nl2sql_train.jsonl")

    # baseline program (before optimization)
    base_program = NL2SQLModule()

    baseline_acc = accuracy_on(train_examples, base_program)
    print(f"[Before Optimization] NL2SQL exact-match accuracy = {baseline_acc:.2f}")

    # Create a dedicated program for the DSPy teleprompter (returns a Prediction object with the `sql` field).

    class TrainableNL2SQL(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(NL2SQL)

        def forward(self, question, schema, constraints_json):
            return self.predict(
                question=question,
                schema=schema,
                constraints_json=constraints_json,
            )

    student = TrainableNL2SQL()

    teleprompter = BootstrapFewShot(
        metric=sql_exact_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=20,
        max_rounds=1,
        max_errors=5,
    )

    print("[Optimization] Running BootstrapFewShot on NL2SQLModule...")
    optimized_program = teleprompter.compile(
        student=student,
        trainset=train_examples,
    )

    # save parameters
    optimized_program.save("optimized_nl2sql.json")
    print("[Optimization] Saved optimized program to optimized_nl2sql.json")

    # measure ass after opt
    optimized_module = NL2SQLModule()
    after_acc = accuracy_on(train_examples, optimized_module)
    print(f"[After Optimization] NL2SQL exact-match accuracy = {after_acc:.2f}")


if __name__ == "__main__":
    main()
