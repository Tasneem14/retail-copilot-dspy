# Hybrid RAG + NL2SQL Agent (LangGraph + DSPy)

## Graph Design 
- The agent uses a **Router → (RAG / SQL / Hybrid)** structure: questions are classified into the correct path using a DSPy classifier.  
- **RAG path:** retrieves top document chunks and passes them to a Planner that extracts KPIs, dates, and categories.  
- **SQL path:** uses an NL2SQL module to generate a single valid SQLite query, followed by an Executor + Repair loop for validation.  
- **Synthesis:** a final Synthesizer merges RAG/SQL results, enforces output types, and returns structured JSON + citations + confidence.

---

## DSPy Optimization
- **Optimized Module:** `NL2SQLModule`  
- **Method:** DSPy `BootstrapFewShot` optimization using a curated NL→SQL dataset.  
- **Metric (SQL correctness):**  
  - **Before:** 3 / 6 correct SQL outputs → **50%**  
  - **After:** 5 / 6 correct SQL outputs → **83%**  
- Improvement reflects better date handling, valid joins, correct table names, and proper KPI aggregation.

---

## Trade-offs & Assumptions
- **Cost of Goods approximation:** margin = revenue − (0.7 × UnitPrice × Quantity) because the schema lacks true cost fields.  
- **Seasonal date inference:** Planner infers windows like “Summer 1997” → `1997-06-01` to `1997-09-01` using heuristics + RAG context.  
- **RAG-only fallback:** some policy questions intentionally avoid SQL and rely solely on retrieved documents for higher accuracy.  
- **Small tuning dataset:** NL2SQL optimization uses a limited handcrafted set, which may not generalize beyond the assignment scope.

