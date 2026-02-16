# Go Online (Web Search / Fetch / Browser)

This repo supports a “Go Online” tool stack so the model can gather information from the internet.

## Web Search (`web_search`)

### What it is
`web_search` is an OpenAI-provided tool that performs a web search and returns a small set of sources/snippets.

This repo supports it in **both** chat engines:
- **Legacy engine** (`ChatSession` / Responses API): passes `{"type":"web_search","search_context_size":...}` in each call.
- **Agents SDK engine** (`AgentsSdkSession` / `openai-agents`): uses `WebSearchTool(search_context_size=...)`.

### Configuration
There are two ways to configure it:

1) **Chat UI Controls menu (recommended)**
- `Enable web search` (on/off)
- `Web search context size`: `low` | `medium` | `high`

2) **Environment variables**
- `OPENAI_API_KEY` – required
- `OPENAI_MODEL` – optional; overrides the model used in the UI session

### Notes / limitations
- Search results can change over time.
- The tool returns *sources + snippets*, not full pages. Use `web_fetch` when you need full page content.
- Costs/latency: search adds latency and may increase token usage due to returned excerpts.

### Recommended prompting pattern
Ask the model to:
1) search,
2) cite 3–5 sources,
3) summarize,
4) extract concrete facts + dates,
5) then answer your question.

Example prompt:
“Search the web for X, list 5 sources with titles and URLs, then summarize and give me the top 3 takeaways.”

