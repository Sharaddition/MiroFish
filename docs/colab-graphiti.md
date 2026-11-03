# Running MiroFish on Colab or Kaggle with Graphiti + FalkorDB

`notebooks/mirofish_graphiti.ipynb` runs the whole pipeline on one free VM: **upload a seed document, write a prompt**,
get the ontology, the knowledge graph, the agents, the simulation and the report. The knowledge graph lives in an
embedded FalkorDB with [Graphiti](https://github.com/getzep/graphiti) on top, so there is no Zep account and no Docker.
Only your language model is called over the network.

Open it in Colab, or import the `.ipynb` into Kaggle (Settings -> Internet **on**; Add-ons -> Secrets). Add the secret
`LLM_API_KEY`, set the model in the Settings cell, and run the cells top to bottom. Step 4 ends with a **smoke test**
that checks your model, your embedder and Graphiti on one sentence before you spend an hour on a full run.

## What is different from the Zep setup

| | Zep Cloud (default) | Graphiti + FalkorDB (`GRAPH_BACKEND=graphiti`) |
|---|---|---|
| Graph storage | Zep's servers | a FalkorDB file on the VM (copied to Drive / `/kaggle/working` at the end) |
| Who reads the seed | Zep's own models | **your** model, through Graphiti: several calls per chunk |
| Waiting at the end of a run | Zep's queue can lag for hours | none: an episode is finished when Graphiti returns |
| Embeddings | Zep | a small local model by default, or any OpenAI-style embeddings API |
| Cost | Zep plan | your model's tokens |

The app talks to either through the same calls: `get_zep_client()` returns a Graphiti-backed object that imitates the
Zep SDK calls MiroFish makes (`app/utils/graphiti_backend.py`). Zep stays the default and its code path is untouched.

## Starting a run

1. Open the notebook (Colab: the badge at its top; Kaggle: import the `.ipynb`, turn Internet on).
2. Add your key as a **Secret** (Colab: key icon in the left bar, *Add new secret*, switch on *Notebook access*):
   `TWITTER_LLM_API_KEY` is the only required one.
3. Edit the **Settings** cell, then run the cells top to bottom: secrets, install, start services, smoke test, your seed
   and prompt, then the six pipeline cells, then the save cell. *Run all* also works; it pauses at the upload dialog.

## Setting environment variables

Three layers, from simplest to most general:

| Where | For | Example |
|---|---|---|
| **Settings** cell | the model for each job, embeddings, rounds, live graph updates | `MAIN_LLM_MODEL_NAME = "some-model"` |
| **Secrets** | keys: `<JOB>_API_KEY`, `EMBEDDING_API_KEY`, `TWITTER_RERANKER_API_KEY` | `REDDIT_LLM_API_KEY` |
| **`EXTRA_ENV`** box (Settings cell) | any other backend setting, one `KEY=VALUE` per line | `LLM_MAX_CONCURRENCY=8` |

`EXTRA_ENV` lines win over the Settings cell. Put keys in Secrets and refer to them with `NAME=secret:SECRET_NAME`
(for example `LLM_FALLBACK_API_KEY=secret:LLM_FALLBACK_API_KEY`), so the key is never typed into the notebook, which is
saved and shared with its text. A line with an *old* name (`LLM_MODEL_NAME`) is ignored by the backend when its new name
(`TWITTER_LLM_MODEL_NAME`) is set from Settings; the cell prints a warning when that happens, so set the new name.
All the settings the backend reads are listed in `.env.example`.

## One model per job

The notebook (and `.env`) can give every job its own model, using the names from `backend/env_aliases.py`. Only the
Twitter model is required; a job left blank follows another one.

| Job | Names | What it runs | Blank means |
|---|---|---|---|
| Twitter model | `TWITTER_LLM_*` (was `LLM_*`) | Twitter agents and the end-of-run poll | required |
| Reddit model | `REDDIT_LLM_*` (was `LLM_BOOST_*`) | Reddit agents | same as Twitter |
| Main model | `MAIN_LLM_*` (was `LLM_PREP_*`) | the ontology, agent personas, the simulation config, the outcome questions, the report and its search tools | same as Twitter |
| Graph model | `GRAPHITI_LLM_*` | reading the seed into the knowledge graph | same as Main |
| Feed ranker | `TWITTER_RERANKER_*` (was `LLM_EMBEDDING_*`) | the hosted twhin-bert behind the Twitter feed | runs on the VM |
| Graph embeddings | `GRAPHITI_EMBEDDING_*` | graph search | a small local model |

Each job takes `API_KEY`, `BASE_URL` and `MODEL_NAME`; in the notebook the keys are Secrets named `<JOB>_API_KEY`.
The backend only starts a separate Reddit or Main model when it has both a key and a model, so the notebook fills in
the key and base URL from the job being followed (`notebooks/mirofish_env.py`, tested). Old `.env` names keep working
and a new name wins when both are set.

## Choosing the model

Graphiti asks the model for structured JSON many times per chunk (entities, facts, duplicates, summaries). A weak or
heavily rate-limited model gives an empty or poor graph, or takes hours. If the smoke test or the graph step struggles,
set `GRAPHITI_LLM_MODEL_NAME` to a stronger model while keeping a cheaper one for the agents. `GRAPHITI_CONCURRENCY`
(default 4) is how many model calls Graphiti makes at once inside one chunk; lower it on a free tier.
`GRAPHITI_EPISODE_WORKERS` (default 1; the notebook sets 4) is how many chunks of the seed are read at the same time. Chunks
are otherwise read one after another, which is slow because each chunk is a chain of dependent model calls. Reading
several together is several times faster, but chunks read together cannot see each other's entities yet, so the same
entity may appear twice (for example "Persistent Systems" and "Persistent"). Keep it within what your provider allows.

`GRAPHITI_DEDUPE` cleans that up once the build finishes, before the graph step is reported complete:

| Value | What it does |
|---|---|
| `off` | Keeps the graph as built. |
| `exact` (default) | Merges entities with the same name (ignoring case and punctuation) and the same type. No model calls. |
| `llm` | Also looks for near-duplicates among the entity names: names that contain each other ("Persistent" inside "Persistent Systems") and names whose embeddings are close ("Infosys Ltd" / "Infosys Limited"). A pair scoring at least `GRAPHITI_DEDUPE_AUTO_SCORE` (default 0.97) is merged without asking; a pair between `GRAPHITI_DEDUPE_ASK_SCORE` (default 0.85) and that is put to the Graphiti model, one call per pair, and only a clear yes merges; anything lower is left alone. `GRAPHITI_DEDUPE_MAX_CALLS` (default 50) caps the model calls. |

Entities are only compared within one type, except that an entity Graphiti left untyped joins the typed one of the same
name (when there is just one). Merging moves every fact and episode link onto the surviving entity, deletes the others and drops facts that are now
exact repeats. Every merge is written to the backend log. A cleanup that fails is logged and skipped: the graph is kept
as built. Acronyms ("SEBI" / "Securities and Exchange Board of India") share no words, so they are not found.

`GRAPHITI_STRUCTURED_OUTPUT=json_object` (the default) works with far more providers than OpenAI's strict
`json_schema` mode.

## Settings

All `GRAPHITI_*` settings are in `.env.example`. The notebook writes them for you.

## Files

| | |
|---|---|
| `notebooks/mirofish_graphiti.ipynb` | the notebook (generated by `notebooks/build_notebook.py`) |
| `notebooks/mirofish_driver.py` | drives the backend's HTTP API stage by stage |
| `notebooks/falkordb_server.py` | keeps the embedded FalkorDB running |
| `backend/constraints-linux.txt` | the package versions known to work together |
| `backend/scripts/graphiti_smoke_test.py` | the smoke test |
| `backend/app/utils/graphiti_backend.py` | the Zep-imitating client and the Graphiti engine |

## Things worth knowing

- **Python 3.12.** Colab's default Python is 3.13; the backend needs 3.11 or 3.12, so the notebook builds its own
  environment with `uv`.
- **CPU-only PyTorch.** The default PyTorch wheel also installs about 5 GB of NVIDIA GPU libraries that a CPU runtime
  never uses. The notebook installs the CPU build first: 1.4 GB on disk instead of 7.3 GB, and about 15 seconds instead
  of about a minute. Nothing in the pipeline computes on a GPU.
- **Downloads that remain.** The packages (1.4 GB), the small local embedding model for graph search (about 130 MB), and
  the twhin-bert model (1.1 GB) that ranks the Twitter feed. That one is fetched only when the feed first needs
  ranking, i.e. once there are more posts than an agent is shown, so a short test run never downloads it and a real run
  does. Set `TWITTER_RERANKER_BASE_URL` to use the hosted ranker and skip it.
- **Pinned versions.** A fresh install picks the newest of everything and breaks (`mcp` 2 no longer has the `FastMCP`
  that `camel-ai` imports). `constraints-linux.txt` holds the versions the backend is tested with. `certifi` and `neo4j`
  are left unpinned on purpose: `graphiti-core` needs newer ones.
- **Free sessions end.** Colab stops idle sessions and caps them at about 12 hours; Kaggle at about 9. Keep the tab
  open. The final cell saves the report, the run folder and the graph database folder. The database snapshots itself
  every 30 seconds while the graph changes (`--save-every`), so even a session that is cut off keeps almost everything
  in `graph_db/` (FalkorDBLite on its own writes only on a clean shutdown).
- **Live graph updates are off by default** (`LIVE_GRAPH_UPDATES`). Turning them on writes every agent action into the
  graph as it happens, which costs several model calls per action.
- **FalkorDBLite** is an embedded build meant for development and prototyping. It is the simplest way to run FalkorDB
  without Docker; for anything long-lived use a real FalkorDB or Neo4j server and point `GRAPHITI_DB_HOST` at it.
