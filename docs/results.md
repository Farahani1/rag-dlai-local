# Results

What changes when the course's RAG chatbot (Week 5) runs on a small local model instead of a hosted one.

> **Status (2026-09-24): final.** Baseline (the course pipeline unchanged, 12 questions × 3 repeats) and three variants (1 repeat each), all with `qwen2.5:1.5b` on the same frozen question set.

## Headline

**On the course's pipeline, a 1.5B local model gets 4 of 12 questions right. Most failures come from one step, routing. Replacing that single LLM call with an embedding lookup doubles the score to 8 of 12 and cuts median latency from 161 s to 36 s.** Fixing the broken filter JSON makes the filters parse, but doesn't improve answers: the filters the small model writes are often wrong.

| Variant | What changes | Correct | Routed correctly | Filter JSON parsed | Fell back to unfiltered | Median latency (min–max) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | nothing: the course pipeline as written | 4 / 12 | 15 / 30 | 0 / 33 | 33 / 33 | 161 s (69–202) |
| parse | strip the code fence; remove `nan` from the filter prompt | 3 / 12 | 5 / 10 | 9 / 11 | 2 / 11 | 152 s (68–190) |
| **router** | **embedding router instead of the routing LLM call** | **8 / 12** | **10 / 10** | 0 / 4 | 4 / 4 | **36 s (4–161)** |
| both | parse + router | 8 / 12 | 10 / 10 | 4 / 4 | 0 / 4 | 34 s (4–174) |

- "Correct" counts questions answered correctly (for the baseline, in the majority of its 3 repeats; all its repeats were identical).
- "Routed correctly" counts runs of the 10 questions with a clear right route: FAQ for the 6 fact questions, Product for the 4 product questions. The two "insufficient context" questions are excluded.
- The filter columns count product-route runs. The router variants send fewer questions down the product route, so their denominators are smaller.
- The latency medians depend on the question mix: the router sends 8 of 12 questions down the FAQ route, which is much faster (see [Where the time goes](#where-the-time-goes)). Product questions take as long as before.

## Before and after

### The router fixes the model's weakest step

The embedding router sends a question to the FAQ branch if its best cosine similarity to one of the 25 FAQ questions is at least 0.4025. That threshold was chosen on a separate dev set ([`benchmark/router_dev.yaml`](../benchmark/router_dev.yaml): 12 FAQ-type and 12 product queries on topics and items the benchmark doesn't use; 24 of 24 correct), and committed before any router run. The benchmark questions played no part in choosing it.

With every question on the right branch:

- 4 wrong answers become right: d2 (45 days), d3 (EUR 9.95), i2 (gift card with a code, valid two years) and n1. On the FAQ branch, the student-discount hallucination is gone: "No, we do not offer a student discount at Fashion Forward Hub."
- n2 stays right, with a better answer: "No, we do not sell bicycles. Our products include clothing, shoes, and accessories…"
- The router makes no LLM call, and it takes milliseconds.

What still fails with the router, and why:

| Question | Answer | Cause |
| --- | --- | --- |
| i1 70-euro order: pay for delivery, and how long? | "No, you do not need to pay extra… 3 to 5 working days and costs EUR 4.95" | Doesn't apply the free-over-60 rule, and contradicts itself: combining two facts is hard for the model |
| m2 How many payment methods? | Lists all six correctly but never says "six" | Counting |
| f2 Black formal shoes under 100 | 12 black men's formal shoes, 5 over 100 | The answer prompt has no prices (the course code) |
| m1 Handbags that are not black | Three black handbags | Negation |

### The parser fix alone doesn't help

With the code fence stripped and `nan` removed, the filter JSON parses in 9 of 11 product-route runs (baseline: 0 of 33). But accuracy doesn't improve (3/12 against 4/12), for two reasons:

1. Parsing doesn't fix routing: FAQ questions still go down the product branch.
2. The filters the model writes are often wrong once they're used. It picks categories that don't exist ("Clothing Set", "Shoes", "Sports"), drops price limits, or for f2 picks "Sports Shoes" for formal shoes, so the filtered search returns the wrong products. In the baseline, failed parsing sent every product question to unfiltered search, which happened to work well for f1 and f3.

n2 is the one answer that got worse: in the baseline it said "No."; with the parsed (invented) filter it asks for more details instead of declining.

In `both`, the four product questions get the same scores as in `router`: the filters now parse and are used (4/4), but f1 and f3 were already right and f2 and m1 fail for other reasons.

### What was tested, and what wasn't

Tested: the parser fix (the JSON parses, but the answers don't improve) and the embedding router (the largest gain). Not tested: Ollama's constrained JSON output (`format: json`), putting prices into the answer prompt, and a larger model. The first two are cheap next variants; the last was dropped as too slow on this laptop.

### Where the router's speed comes from: the prompt cache

The FAQ answer prompt carries all 25 FAQ entries: about 1,330 tokens. On this CPU, `qwen2.5:1.5b` reads a prompt at about 20 tokens per second, so a cold FAQ answer takes about 65 s just to read its prompt. Ollama keeps the most recent prompt's processed state and reuses any shared beginning, but it keeps only one. Measured directly:

| Call | Prompt tokens | Time reading the prompt | Total |
| --- | --- | --- | --- |
| FAQ prompt for d2, cold | 1,328 | 64.0 s | 65.4 s |
| FAQ prompt for d3, straight after d2 | 1,324 | 0.6 s | 3.7 s |
| FAQ prompt for d3, after one unrelated prompt | 1,324 | 66.0 s | 68.6 s |

In the course pipeline every question makes its routing call first, and it has a different prompt, so the FAQ answer always starts cold. With the embedding router, consecutive FAQ questions share the FAQ prompt and answer in 3–7 s; only the first FAQ question, and one following product questions, pays the 65 s. In a live chat the gain depends on the order of questions; the benchmark's order is fixed and the same for all variants.

The same effect is why product questions don't get faster: their route, task-type, filter and answer prompts all differ, so every one of them starts cold.

## Question set

Twelve questions in [`benchmark/questions.yaml`](../benchmark/questions.yaml) (version 1), committed before any model was run and not changed since.

| Category | Count | What it probes |
| --- | --- | --- |
| Direct | 3 | One FAQ entry holds the answer |
| Indirect | 2 | The answer combines two FAQ facts |
| Filter | 3 | The model must turn the request into a metadata filter (colour, gender, type, season, price) |
| Limitation | 2 | Negation ("not black") and counting |
| Insufficient context | 2 | The shop doesn't cover it; the right answer is to say so |

Scoring is rule-based ([`benchmark/scoring.py`](../benchmark/scoring.py)): FAQ answers must contain the required facts; product answers must name enough catalogue product IDs, and every product named must match the request's attributes; "insufficient" answers must decline. Where the rules misjudged an answer, the results CSV's `manual_override` column corrects it (see [Manual corrections](#manual-corrections)).

## Method

- Every question goes through the notebook's own W5 functions, unchanged: `answer_query(question, simplified=False)`, then the final generation call. [`benchmark/run.py`](../benchmark/run.py) loads them from the notebook mirror and only wraps the LLM call, so every call uses the model under test at temperature 0.
- The models are loaded with a warm-up call before timing starts.
- Hardware: Intel Core i3-1005G1 (10th gen, 4 threads), 11.8 GB RAM, Windows 11, CPU only. Model digest, OS, Python version and start time are in each CSV's `#` header lines.
- One model: `qwen2.5:1.5b`. The baseline ran 3 repeats; each variant ran 1, because the baseline's repeats were identical (next point). A 4B model was planned as an upper bound and dropped: at these latencies a full run would take several hours on this laptop.
- Variants (`--variant parse|router|both`, [`benchmark/variants.py`](../benchmark/variants.py)) patch the loaded notebook functions; the notebook itself is unchanged.
- **Determinism:** all 12 questions produced word-for-word identical answers in all 3 repeats. Only latency varies between repeats.

## Baseline in detail

The rest of this page describes the baseline run: the course pipeline as written.

## Per question

| Question | Category | Route taken | Result | Why |
| --- | --- | --- | --- | --- |
| d1 Customer support hours | direct | FAQ | ✅ | Exact hours and days from the FAQ |
| d2 Days to return an item | direct | Product | ❌ | Misrouted; answered with a product ID ("49074") |
| d3 Express delivery cost | direct | Product | ❌ | Misrouted; "none of the products have a delivery cost" |
| i1 Delivery for a 70-euro order | indirect | Product | ❌ | Misrouted; asked for the country instead |
| i2 Gift card with a discount code, validity | indirect | Product | ❌ | Misrouted; invented "valid for 12 months" (FAQ: two years) |
| f1 Three blue T-shirts for men | filter | Product | ✅ | Three blue men's T-shirts, although the filter failed |
| f2 Black formal shoes for men under 100 | filter | Product | ❌ | 12 black men's formal shoes, but 5 cost over 100 (see below) |
| f3 Two red summer dresses for women | filter | Product | ✅ | Two red women's summer dresses |
| m1 Three women's handbags, not black | limitation | Product | ❌ | All three are black: negation ignored |
| m2 How many payment methods | limitation | Product | ❌ | Misrouted; listed dresses instead |
| n1 Student discount? | insufficient | Product | ❌ | Hallucinated: "Yes, we do offer a student discount" |
| n2 Do you sell bicycles? | insufficient | Product | ✅ | "No." |

## Why it fails

Four causes, in order of how many answers they cost:

1. **Routing (the model).** 5 of the 6 questions the FAQ answers (d2, d3, i1, i2, m2) and both "insufficient" questions were routed to the product branch; only d1 went to FAQ. The product branch never sees the FAQ, so these answers were wrong or evasive. **This alone accounts for 5 of the 8 wrong answers.**
2. **The filter JSON never parses (the course's code and data, not the filter's content).** In all 33 product-route runs retrieval fell back to unfiltered semantic search. Two separate causes:
   - In 24 of the 33 outputs the model wrapped the JSON in a Markdown code fence (```` ```json ````), which the notebook's `parse_json_output` doesn't strip. Stripping the fence alone makes 18 of the 33 parse. Their content is sensible: for "three blue T-shirts for men", `gender: Men, articleType: T-shirts, baseColour: Blue`.
   - 15 outputs contain a bare `nan`, which isn't valid JSON. The prompt's list of allowed colours includes `nan`, because 15 catalogue products have no colour, and the model copies it. This happened on the non-product questions (d2, d3, i2, n1, n2), where there was no colour to choose.
3. **The answer prompt has no prices (the course's code).** `generate_items_context` passes each product's name, type, colour, season and so on, but not its price. In f2, unfiltered search still found 12 black men's formal shoes, but the model had no way to honour "under 100", and 5 of the 12 cost more.
4. **Negation (the model and the retrieval).** m1 asked for handbags that are not black; all three answers are black. Semantic search on "not black" retrieves black bags, and the model doesn't filter them out.

What the small model does well: when the right context reaches it, its answer is accurate and to the point (d1, f1, f3), and it declined the bicycle question correctly. Its one clear hallucination is the student discount (n1).

## One failure, traced end to end

f2, repeat 1: *"I need black formal shoes for men that cost less than 100."* The full trace, including the complete prompt, is in [`benchmark/results/trace_f2_r1.md`](../benchmark/results/trace_f2_r1.md). It was rebuilt with [`benchmark/trace.py`](../benchmark/trace.py) from the recorded output, without calling the model again.

1. **Route:** Product (correct).
2. **Filter prompt (4,238 characters):** the allowed values are pasted in as a Python dict of sets, not JSON. The colour list contains `nan` **15 times**: each product without a colour adds its own `nan`, because in Python `nan != nan`, so a set never merges them.
3. **Raw model output:**

   ````text
   ```json
   {
       "gender": ["Men"],
       "masterCategory": ["Footwear"],
       "articleType": ["Shoes"],
       "baseColour": ["Black"],
       "price": {"min": 0, "max": "inf"},
       "usage": ["Formal"],
       "season": ["All seasons"]
   }
   ```
   ````

4. **Parse:** `parse_json_output` fails on the first character, the code fence: `JSON parsing failed: Expecting value: line 1 column 1 (char 0)`. No filters are built.
5. **Retrieval:** falls back to unfiltered semantic search over the whole catalogue.
6. **Final answer:** 12 product IDs. All 12 are black men's formal shoes, so the unfiltered search did well, but 5 cost more than 100, and the answer prompt shows no prices. Scored wrong.

Even without the fence, this filter wouldn't have helped: the model dropped the price limit (`"max": "inf"` for "less than 100") and chose `"Shoes"`, which isn't in the allowed list (the catalogue says "Formal Shoes"). The parser fix recovers the JSON, but not the filter's quality. That is why the before/after experiments measure the fixes one at a time.

## Where the time goes

The pipeline makes 2 LLM calls for an FAQ question (route, answer) and 4 for a product question (route, task type, filter JSON, answer). Ollama's server log records each call's duration; [`benchmark/step_times.py`](../benchmark/step_times.py) matches the calls to the questions and splits each question's latency into steps. The per-question result is saved in `benchmark/results/qwen2.5-1.5b_2026-09-24_steps.csv`, because Ollama rotates its log.

All 36 runs (3 FAQ-routed, 33 product-routed):

| Step | FAQ route (median s) | Product route (median s) | Share of a product question |
| --- | --- | --- | --- |
| Route (LLM) | 7.9 (6–8) | 7.2 (6–12) | 4% |
| Task type (LLM) | — | 6.1 (5–11) | 4% |
| Filter JSON (LLM) | — | 67.0 (58–82) | 42% |
| Final answer (LLM) | 76.0 (58–77) | 71.0 (10–109) | 44% |
| Everything else (embedding, Chroma, Python) | 4.2 (4–5) | 9.4 (8–12) | 6% |
| **Total** | 89.1 (69–89) | 161.3 (100–202) |  |

Cells show median (min–max). One run (m2, repeat 3) is left out of the total and "everything else": the laptop went to sleep during it (see [Manual corrections](#manual-corrections)). Its LLM step times are unaffected and included.

What this shows:

1. **Two long-prompt LLM calls take about 90% of the time.** The filter prompt lists every allowed value of every product field, and the answer prompt carries 20 retrieved products. On a CPU, reading a long prompt is the expensive part. Retrieval itself (embedding the query and searching Chroma) takes a few seconds.
2. **The filter step costs 42% of a product question and never produced a usable filter.** As the pipeline stands, removing the step would make product questions about 40% faster with the same answers; fixing it would make that minute useful.
3. **Routing is cheap (about 7 s).** Its problem is accuracy, not time.

Caveats: the log rounds calls longer than a minute to whole seconds, and the matching assumes nothing else used Ollama during the run.

## Manual corrections

All in `benchmark/results/qwen2.5-1.5b_2026-09-24.csv`, applied to all 3 repeats of the question:

| Row(s) | Column | Value | Reason |
| --- | --- | --- | --- |
| n2 | `manual_override` | y | "No." to "Do you sell bicycles?" is correct; the abstain rule's phrase list has no bare "no". |
| m2 | `manual_override` | n | The answer lists dresses instead of counting payment methods; the fact rule matched "6" inside product IDs such as 59963. |
| m2, repeat 3 | `latency_excluded` | laptop slept 19:13–20:13 | Windows went to sleep during this question (System events 42 and 107). Its recorded latency (3,782 s) includes the hour asleep. Accuracy is unaffected. |

Both scoring gaps are rule bugs, not question changes. A scorer fix (whole-number matching for numeric facts; a bare "no" as an abstain answer) would apply to every run, including this baseline.

## How to reproduce

```bash
python benchmark/run.py --model qwen2.5:1.5b --repeats 3                     # baseline, ~1.5 h on the i3
python benchmark/run.py --model qwen2.5:1.5b --repeats 1 --variant router    # or parse / both
python benchmark/tune_router.py                                              # how the router threshold was chosen
python benchmark/summarize.py benchmark/results/*.csv                        # headline table
python benchmark/step_times.py benchmark/results/qwen2.5-1.5b_2026-09-24.csv     # time per step (needs the Ollama server log)
```

Requires Ollama with the model pulled and the W5 Chroma collections populated (see [setup.md](setup.md)). Don't run anything else against Ollama during a run, or the step timings can't be matched; and disable sleep, or a run can stall for as long as the laptop sleeps.
