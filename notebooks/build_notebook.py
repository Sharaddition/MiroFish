"""Writes notebooks/mirofish_graphiti.ipynb. Run:  python notebooks/build_notebook.py

The notebook is generated so its cells stay reviewable as plain Python here, and so one edit updates the notebook
for both Colab and Kaggle.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OPEN_IN_COLAB = "https://colab.research.google.com/github/Sharaddition/MiroFish/blob/feat/graphiti-colab-notebook/notebooks/mirofish_graphiti.ipynb"

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(f"""
# MiroFish on Colab / Kaggle, with Graphiti + FalkorDB

Upload a **seed document** and write a **prompt**; this notebook builds the knowledge graph, creates the agents, runs
the social simulation and writes the report, **entirely on this VM**. The graph lives in an embedded FalkorDB with
[Graphiti](https://github.com/getzep/graphiti) on top (no Zep account, no Docker). Only your language model is called
over the network.

[![Open in Colab]({"https://colab.research.google.com/assets/colab-badge.svg"})]({OPEN_IN_COLAB})

**You need**
1. An API key for an OpenAI-compatible language model. Add it as a **Secret** named `LLM_API_KEY`
   (Colab: the key icon in the left bar, switch on *Notebook access*. Kaggle: *Add-ons → Secrets*, and switch **Internet on**).
2. A model that follows JSON instructions well. The graph step asks it for structured output many times; a weak or
   rate-limited free model is the most common reason for a poor or slow graph. Step 4 tests yours first.

**How long it takes** (free CPU runtime, 12 GB RAM): install about 6 minutes; graph from a 5-page seed 5 to 30 minutes
depending on the model's speed; persona generation about 2 minutes per 10 agents; the simulation 30 to 90 minutes for 24
rounds; the report 5 to 15 minutes. Keep the browser tab open: free sessions stop when idle.

Run the cells top to bottom. If a cell fails, fix the cause and re-run **that** cell.
""")

md("## 1. Settings")
code('''
#@title Settings (edit, then run)
# --- where the code comes from -------------------------------------------------------------------------------------
REPO_URL = "https://github.com/Sharaddition/MiroFish.git"  #@param {type:"string"}
BRANCH = "feat/graphiti-colab-notebook"                      #@param {type:"string"}

# --- the language model for agents, personas, config and the report (any OpenAI-compatible API) -----------------------
LLM_BASE_URL = "https://openrouter.ai/api/v1"               #@param {type:"string"}
LLM_MODEL_NAME = "google/gemini-2.0-flash-001"              #@param {type:"string"}

# --- the model Graphiti uses to read the seed (blank = same as above). It must follow JSON instructions reliably. -------
GRAPHITI_LLM_BASE_URL = ""                                   #@param {type:"string"}
GRAPHITI_LLM_MODEL_NAME = ""                                 #@param {type:"string"}

# --- embeddings for graph search: "local" runs a small model on this VM (nothing to configure); "api" uses a service ----
EMBEDDING = "local"                                          #@param ["local", "api"]
EMBEDDING_BASE_URL = ""                                      #@param {type:"string"}
EMBEDDING_MODEL_NAME = ""                                    #@param {type:"string"}
EMBEDDING_DIM = 768                                          #@param {type:"integer"}

# --- the run ------------------------------------------------------------------------------------------------------------
MAX_ROUNDS = 24                                              #@param {type:"integer"}
LIVE_GRAPH_UPDATES = False                                   #@param {type:"boolean"}
# LIVE_GRAPH_UPDATES=True also writes every agent action into the graph as it happens: richer reports, but each
# action costs extra model calls and the end of the run waits for them. Leave it off for a first run.
GRAPHITI_CONCURRENCY = 4                                     #@param {type:"integer"}
# parallel model calls Graphiti makes; lower it (2) if your provider rate-limits you
''')

md("## 2. Secrets and folders")
code('''
import os, sys, subprocess, getpass, shutil, time, json, textwrap

ON_KAGGLE = os.path.exists("/kaggle/working")
WORK = "/kaggle/working" if ON_KAGGLE else "/content"
REPO = f"{WORK}/mirofish"
VENV = f"{WORK}/venv"
LOGS = f"{WORK}/logs"
os.makedirs(LOGS, exist_ok=True)

def secret(name, required=False):
    """Colab Secrets, then Kaggle Secrets, then an environment variable, then (if required) a hidden prompt."""
    try:
        from google.colab import userdata
        value = userdata.get(name)
        if value: return value
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret(name)
        if value: return value
    except Exception:
        pass
    if os.environ.get(name): return os.environ[name]
    if required:
        return getpass.getpass(f"{name} (not found in Secrets; paste it, it is not saved): ")
    return ""

LLM_API_KEY = secret("LLM_API_KEY", required=True)
GRAPHITI_LLM_API_KEY = secret("GRAPHITI_LLM_API_KEY") or LLM_API_KEY
EMBEDDING_API_KEY = secret("EMBEDDING_API_KEY") if EMBEDDING == "api" else ""
GITHUB_TOKEN = secret("GITHUB_TOKEN")   # only needed if the repository is private
print("platform:", "Kaggle" if ON_KAGGLE else "Colab / other", "| work folder:", WORK, "| python:", sys.version.split()[0])
''')

md("## 3. Get the code and install (about 6 minutes)")
code('''
def run(cmd, cwd=None, quiet=False, timeout=3600):
    """Run a shell command, stream its output, raise with the tail of the output if it fails."""
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail, started = [], time.time()
    for line in proc.stdout:
        tail = (tail + [line])[-40:]
        if not quiet: print(line, end="")
        if time.time() - started > timeout:
            proc.kill(); raise TimeoutError(cmd)
    if proc.wait() != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}\\n" + "".join(tail))

# 1) the code: an uploaded tarball wins (for testing a change that is not pushed yet), otherwise git
if not os.path.exists(f"{REPO}/backend/run.py"):
    tarball = f"{WORK}/mirofish_src.tar.gz"
    if os.path.exists(tarball):
        os.makedirs(REPO, exist_ok=True); run(f"tar xzf {tarball} -C {REPO}", quiet=True)
    else:
        url = REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@") if GITHUB_TOKEN else REPO_URL
        run(f"git clone --depth 1 --branch {BRANCH} {url} {REPO}", quiet=True)
print("code ready:", os.path.exists(f"{REPO}/backend/run.py"))

# 2) Python 3.12 environment (the backend does not run on Colab's default Python 3.13), pinned to known-good versions
run("pip -q install uv", quiet=True)
run(f"uv venv --clear --python 3.12 {VENV}", quiet=True)
PIP = f"uv pip install --python {VENV}/bin/python -c {REPO}/backend/constraints-linux.txt"
run(f"{PIP} -r {REPO}/backend/requirements.txt", quiet=True)
run(f"{PIP} 'graphiti-core[falkordb]==0.31.2' falkordblite", quiet=True)
run(f"{VENV}/bin/python -c \\"import oasis, graphiti_core, redislite, flask; print('install ok')\\"")
''')

md("""
## 4. Start the graph database and the backend, then test your model

The smoke test creates a tiny graph with **your** model and embedder, extracts one sentence, and searches it. It
takes a minute and tells you what to change if anything is wrong, before you spend an hour on a full run.
""")
code('''
# .env for the backend (it reads this file; keys stay on this VM)
env = {
    "GRAPH_BACKEND": "graphiti",
    "ZEP_GRAPH_MEMORY_UPDATE": "true" if LIVE_GRAPH_UPDATES else "false",
    "LLM_API_KEY": LLM_API_KEY, "LLM_BASE_URL": LLM_BASE_URL, "LLM_MODEL_NAME": LLM_MODEL_NAME,
    "GRAPHITI_DB_PORT": "6390", "GRAPHITI_CONCURRENCY": str(GRAPHITI_CONCURRENCY),
    "GRAPHITI_STATE_DIR": f"{WORK}/graphiti_state",
    "GRAPHITI_LLM_API_KEY": GRAPHITI_LLM_API_KEY,
    "GRAPHITI_LLM_BASE_URL": GRAPHITI_LLM_BASE_URL or LLM_BASE_URL,
    "GRAPHITI_LLM_MODEL_NAME": GRAPHITI_LLM_MODEL_NAME or LLM_MODEL_NAME,
    "FLASK_PORT": "5001", "FLASK_HOST": "127.0.0.1",
    "LLM_MODEL_TIMEOUT": "180", "LLM_MODEL_MAX_RETRIES": "1",
}
if EMBEDDING == "api":
    env.update(GRAPHITI_EMBEDDING_BASE_URL=EMBEDDING_BASE_URL, GRAPHITI_EMBEDDING_MODEL_NAME=EMBEDDING_MODEL_NAME,
               GRAPHITI_EMBEDDING_API_KEY=EMBEDDING_API_KEY, GRAPHITI_EMBEDDING_DIM=str(EMBEDDING_DIM))
with open(f"{REPO}/.env", "w") as f:
    f.write("\\n".join(f"{k}={v}" for k, v in env.items() if v != "") + "\\n")

def start(name, cmd, cwd):
    log = open(f"{LOGS}/{name}.log", "w")
    return subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

# stop anything left from an earlier run of this cell
for p in globals().get("SERVICES", []):
    try: p.terminate()
    except Exception: pass
subprocess.run("pkill -x redis-server", shell=True)
time.sleep(1)

SERVICES = [
    start("falkordb", f"{VENV}/bin/python {REPO}/notebooks/falkordb_server.py --path {WORK}/graph_db/mirofish.db --port 6390", REPO),
]
time.sleep(4)
SERVICES.append(start("backend", f"{VENV}/bin/python {REPO}/backend/run.py", f"{REPO}/backend"))

sys.path.insert(0, f"{REPO}/notebooks")
from mirofish_driver import MiroFish
mf = MiroFish("http://127.0.0.1:5001")
mf.wait_for_backend(180)
print("graph database + backend are up (logs in", LOGS + ")")
''')
code('''
#@title Smoke test: your model + embedder + Graphiti (run before the long steps)
run(f"{VENV}/bin/python {REPO}/backend/scripts/graphiti_smoke_test.py", cwd=f"{REPO}/backend")
''')

md("## 5. Your seed and your prompt")
code('''
#@title The prompt: what do you want to predict or understand?
PROMPT = """
Simulate how retail investors, institutional investors and the financial press react over the next five trading
sessions to the company's announcement in the attached document, and what stance each group ends up with.
"""

SEED_DIR = f"{WORK}/seed"
os.makedirs(SEED_DIR, exist_ok=True)
seed_files = []
try:
    from google.colab import files
    print("Choose your seed file(s): .txt, .md or .pdf")
    for name, data in files.upload().items():
        with open(f"{SEED_DIR}/{name}", "wb") as f: f.write(data)
except ImportError:
    pass   # Kaggle / elsewhere: put files in /kaggle/working/seed, or attach a dataset (read below)
for root in (SEED_DIR, "/kaggle/input"):
    for dirpath, _, names in os.walk(root) if os.path.exists(root) else []:
        seed_files += [os.path.join(dirpath, n) for n in names if n.lower().endswith((".txt", ".md", ".pdf"))]
print("seed files:", seed_files or "NONE - upload a file or place it in " + SEED_DIR)
assert seed_files, "no seed file found"
''')

md("""
## 6. Run the pipeline

Each cell is one stage. They print progress, and each can be re-run on its own if it fails.
""")
code('''
# 6a. The model reads your seed and your prompt and designs the entity and relationship types
project_id = mf.generate_ontology(seed_files, PROMPT, project_name="colab run")
''')
code('''
# 6b. Graphiti reads the whole seed and builds the knowledge graph (the slowest model-heavy stage)
graph_id = mf.build_graph(project_id)
''')
code('''
# 6c. One persona per entity in the graph, plus the simulation settings
simulation_id = mf.create_and_prepare(project_id, graph_id)
''')
code('''
# 6d. The social simulation: all agents post, comment and react on two simulated platforms
state = mf.run(simulation_id, max_rounds=MAX_ROUNDS, live_graph_updates=LIVE_GRAPH_UPDATES)
state.get("runner_status"), state.get("current_round"), state.get("total_rounds")
''')
code('''
# 6e. The report agent searches the graph and the simulation, then writes the report
report_md = mf.report(simulation_id, allow_partial_graph=bool(state.get("graph_sync_pending")))
open(f"{WORK}/report.md", "w").write(report_md)
from IPython.display import Markdown, display
display(Markdown(report_md))
''')

md("## 7. Save your results")
code('''
# Everything worth keeping: the report, the run folder (agents, posts, config) and the graph database file.
OUT = f"{WORK}/mirofish_results"
os.makedirs(OUT, exist_ok=True)
shutil.copy(f"{WORK}/report.md", f"{OUT}/report.md")
sim_dir = f"{REPO}/backend/uploads/simulations/{simulation_id}"
shutil.make_archive(f"{OUT}/simulation", "zip", sim_dir)
for p in SERVICES: p.terminate()                      # stops the services; the database writes its file on the way out
for p in SERVICES:
    try: p.wait(timeout=60)
    except Exception: p.kill()
if os.path.isdir(f"{WORK}/graph_db"):
    shutil.copytree(f"{WORK}/graph_db", f"{OUT}/graph_db", dirs_exist_ok=True)   # the knowledge graph
print(os.listdir(OUT))

if not ON_KAGGLE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        shutil.copytree(OUT, "/content/drive/MyDrive/mirofish_results", dirs_exist_ok=True)
        print("copied to Google Drive: MyDrive/mirofish_results")
    except Exception as error:
        print("Drive copy skipped (", error, ") - download the files from the left file panel instead:", OUT)
else:
    print("On Kaggle, everything in /kaggle/working is saved with the notebook version (Save & Run All, or Save Version).")
''')

md("""
## Troubleshooting

| What you see | Cause and fix |
|---|---|
| Step 3 fails in `pip`/`uv` | Internet is off (Kaggle: Settings → Internet on). Re-run the cell. |
| Smoke test `[FAIL] ... 401` | The key in Secrets is wrong or not shared with this notebook. |
| Smoke test `[FAIL] ... json` | The model cannot give structured output. Set `GRAPHITI_LLM_MODEL_NAME` to a stronger model. |
| Graph step is very slow or stalls | Rate limits. Lower `GRAPHITI_CONCURRENCY` to 2, or use a faster model for Graphiti. |
| `6b` finishes but the graph is nearly empty | The model returned no entities: use a stronger `GRAPHITI_LLM_MODEL_NAME`. |
| Backend did not come up | Read `logs/backend.log` in the file panel. |
| Session disconnected mid-run | Re-run steps 2 to 4; the simulation can be resumed from the web API (`/api/simulation/resume`). |
""")


def main() -> None:
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": "MiroFish with Graphiti + FalkorDB", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = os.path.join(HERE, "mirofish_graphiti.ipynb")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(notebook, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
