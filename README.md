# Do LLMs Have Consistent Personalities?

Empirical study of Big Five personality profiles across five frontier LLMs, with deflection analysis and behavioral reframing.

**Models:** GPT-5.4, Claude Sonnet 4.6, Gemini Flash Lite, Kimi K2.5, Qwen 3.5 397B

## Files

| File | Purpose |
|---|---|
| `collect.py` | Data collection (smoke or full) |
| `analyze.py` | Score computation + deflection analysis |
| `write_paper.py` | Generates paper.html |

## Instruments

- **BFI-10**: Big Five Inventory short form (1–5 Likert)
- **TIPI**: Ten-Item Personality Inventory (1–7 Likert)
- **Behavioral**: Custom forced-choice A/B pairs (avoids AI deflection)
- **Deflection study**: Direct experience questions, measures refusal rates

## Two conditions per instrument

- **Instructed**: System prompt tells model to answer directly without AI caveats
- **Free**: No system prompt, model responds however it chooses

## Usage

```bash
export OPENROUTER_API_KEY=your_key
python collect.py --smoke   # verify pipeline
python collect.py           # full overnight run
python analyze.py           # compute scores
python write_paper.py       # generate paper.html
```
