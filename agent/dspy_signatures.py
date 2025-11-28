# agent/dspy_signatures.py

import dspy


class RouteQuestion(dspy.Signature):
    """Decide whether to use RAG only, SQL only, or hybrid (RAG + SQL)."""

    question: str = dspy.InputField()
    route: str = dspy.OutputField(
        desc="One of: rag, sql, hybrid. Prefer 'rag' for pure policy/docs, "
        "'sql' for pure numeric DB questions, 'hybrid' when docs define KPIs/dates and DB has numbers."
    )


class PlanQuestion(dspy.Signature):
    """
    Extract high-level constraints: date ranges, KPI name, category, entities.
    """

    question: str = dspy.InputField()
    top_docs: str = dspy.InputField(
        desc="Concatenated relevant doc chunks from RAG search."
    )
    constraints_json: str = dspy.OutputField(
        desc=(
            "A compact JSON object string like "
            '{"kpi": "...", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", '
            '"category": "..."}'
        )
    )


class NL2SQL(dspy.Signature):
    """
    Generate a single valid SQLite query using the live schema.

    Rules:
    - Use double quotes around the table name "Order Details".
    - Always join through Orders to access dates:
        FROM "Order Details" od
        JOIN Orders o ON o.OrderID = od.OrderID
        JOIN Products p ON p.ProductID = od.ProductID
        JOIN Categories c ON c.CategoryID = p.CategoryID
    - Use date ranges, e.g.:
        o.OrderDate >= '1997-06-01' AND o.OrderDate < '1997-07-01'
      instead of YEAR(), MONTH() etc.
    - For gross margin KPIs, use:
        revenue = SUM(od.UnitPrice * od.Quantity * (1 - od.Discount))
        cost    = SUM(od.UnitPrice * od.Quantity * 0.7)
        margin  = revenue - cost
    - Only use existing tables/columns from the schema.
    - Return a single SELECT statement, no comments, no CTEs.
    """

    question: str = dspy.InputField()
    schema: str = dspy.InputField(
        desc="SQLite schema from PRAGMA, including tables Orders, 'Order Details', Products, Customers, Categories."
    )
    constraints_json: str = dspy.InputField(
        desc="JSON with inferred constraints (dates, KPI, category, etc.)."
    )
    sql: str = dspy.OutputField(
        desc="A single SELECT statement compatible with SQLite, no comments."
    )



class SynthesizeAnswer(dspy.Signature):
    """
    Convert SQL results + docs into final answer text that will later be parsed to the requested format.
    """

    question: str = dspy.InputField()
    format_hint: str = dspy.InputField()
    top_docs: str = dspy.InputField(
        desc="Concatenated top RAG chunks with their IDs."
    )
    sql: str = dspy.InputField(
        desc="The final executed SQL query, or empty if not used."
    )
    sql_rows_json: str = dspy.InputField(
        desc="A short JSON-like representation of SQL rows/columns, for the model to reason about."
    )
    answer_text: str = dspy.OutputField(
        desc=(
            "Return ONLY the value for final_answer, serialized as JSON that exactly"
            " matches format_hint (e.g., 14, 123.45, {\"category\":\"Beverages\",\"quantity\":42}, "
            "[{\"product\":\"Chai\",\"revenue\":123.45}, ...])."
        )
    )
    explanation: str = dspy.OutputField(
        desc="One or two short sentences explaining how the answer was obtained."
     )
    confidence: float = dspy.OutputField(
        desc="A confidence score between 0.0 and 1.0."
    )
