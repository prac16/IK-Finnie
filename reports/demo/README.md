# Finnie demo

Final video: `../finnie_demo_5min.mp4` (1920×1080, narrated, 5 minutes).
Selectable English captions are embedded; `../finnie_demo_5min.srt` is the sidecar.
`transcript.md` contains the narration. Captions use approximate sentence timings.

This is an edited narrated walkthrough of captured prototype views, with architecture cards and a synthesized macOS Samantha voice. It is not an uninterrupted screen recording.

| Time | Topic |
|---|---|
| 00:00–01:00 | System design, LangGraph, RAG and privacy boundaries |
| 01:00–01:20 | Financial knowledge chat |
| 01:20–01:45 | Portfolio metrics |
| 01:45–02:10 | Market metrics |
| 02:10–02:40 | Goal projection algorithms |
| 02:40–03:00 | Saved profile inputs survive an actual server restart |
| 03:00–04:00 | Failure handling, including a captured partial-result fallback |
| 04:00–05:00 | Future improvements |

Persistence verification used only educational samples. In the UI, a local `Finnie Demo` profile was created, Tech Heavy holdings and the House deposit goal were explicitly saved, the Streamlit server process was stopped and relaunched, and the browser was reloaded. Selecting the same profile exposed both saved inputs. Both were restored, and fresh analyses were executed. The portfolio source was `saved_holdings.csv`. Session reports were not persisted. The current profile schema stores one selected holdings snapshot and one selected goal plan; it does not archive every independently created goal.

Market and goals views show generated LLM explanations. The recorded portfolio fallback retained calculated metrics and a local summary, with no provider error text displayed. Failure diagrams describe implemented recovery policies; no API outage was induced for the recording.

Verification: 221 tests passed before assembly. Video/audio streams, chapter metadata, captions, duration and decoding were checked separately in `verification.txt`.
