# Behavior v2, seeds and ensembles

This page describes what the `feat/behavior-and-ensembles` branch changes, how to use it, and, just as
important, what its numbers do and do not mean. The design rationale and the acceptance criteria live in
[`plans/PLAN-behavior-wiring-and-ensembles.md`](plans/PLAN-behavior-wiring-and-ensembles.md).

**In short**

1. The agent and platform settings that Step 2 shows now actually change how a simulation runs (*behavior v2*).
2. Every run has a **seed** that makes its *schedule* reproducible.
3. You can run one prepared scenario **N times** with different seeds (an *ensemble*), ask every simulated agent
   a few questions at the end of each run, and look at the **spread** of the answers instead of a single run.
4. A report can be written from an ensemble, and it quotes ranges instead of single numbers.

> **Read this before you trust a number.** An ensemble gives you distributions of *what simulated participants
> said* across N stochastic runs. It is **not** a calibrated probability that the real-world event happens.
> MiroFish has not been backtested for any domain. More runs make the spread more stable; they do not make the
> simulation more accurate.

---

## 1. Behavior v2

New simulations are prepared with `behavior_version: 2` in `simulation_config.json`. Simulations prepared
before this branch have no such field and keep the legacy behavior (v1) exactly as before.

| Setting (generated, editable data) | What it does in a v2 run |
|---|---|
| `agent_configs[].stance`, `sentiment_bias` | A short *starting disposition* paragraph is appended to the agent's profile ("you currently lean supportive, your tone tends to be slightly negative ... this is your starting view, not a script"). Written in the simulation's language. |
| `activity_level`, `posts_per_hour`, `comments_per_hour` | Per-round activation probability `p = activity_level × (1 − e^(−(posts+comments) × minutes_per_round / 60)) × time-of-day multiplier`, capped at 0.95. The rates decide *how often* an agent is woken; the LLM still chooses what it then does. |
| `active_hours` | An agent can only be woken during its active hours (as before). |
| `time_config.agents_per_hour_min/max` | An upper bound on how many agents are woken in a round (as in v1, scaled by the peak/off-peak multiplier). |
| `time_config` peak, off-peak, **morning and work** multipliers | Time-of-day multiplier. v1 silently ignored the morning and work multipliers. |
| `response_delay_min/max` | Reaction latency. When something is posted (the initial posts, each scheduled event) every agent draws a delay in simulated minutes and cannot be woken before it has passed. |
| `influence_weight`, `stance`, platform `echo_chamber_strength` | The **seeded follow graph** each run starts with: `P(i follows j) = 0.3 × influence(j)/max_influence × (1 + echo·[same side] − echo·[opposite sides])`, clamped to 0.9. `echo_chamber_strength` defaults to 0.5 (Twitter) and 0.6 (Reddit) when the platform config has none. |
| `event_config.topic` | The topic named in the starting-disposition paragraph. |
| `event_config.initial_posts` | Posted when the run starts (as before). |
| `event_config.scheduled_events` | **User-authored** events, see below. |
| `event_config.narrative_direction` | **Never given to any agent.** It is the generator's guess at the outcome; feeding it back would make the forecast circular. A report treats it as a hypothesis to confirm or reject. |

The original profile files are never modified. A run writes `twitter_profiles.effective.csv` and
`reddit_profiles.effective.json` next to them (the originals plus the starting-disposition paragraph) and the
simulation reads those. The UI and `/profiles` keep showing the originals.

The generator also stops emitting platform fields OASIS never used (`recency_weight`, `popularity_weight`,
`relevance_weight`, `viral_threshold`), and sanitises what the LLM returns (clamping ranges, filling defaults).

### Scheduled events

Step 2 has a *Scheduled events* editor. An event has a simulation hour, a poster (a specific agent or an
agent type), and the text to post. At the start of the round that contains that hour the post is published as
that agent, other agents can see and react to it, and it is logged as `injected`. The generator may *suggest*
events; suggestions are saved disabled and only run when you accept them. Events can only be edited while the
simulation is ready (not started): `PUT /api/simulation/<id>/scheduled-events`.

### Switching it off

Set `BEHAVIOR_V2_ENABLED=false` in `.env` to force the legacy behavior for every simulation without touching any
config. It is an emergency switch and defaults to on.

### Preparing a simulation with a slow model

Step 2 asks the model for the time settings, the opening posts and the agent settings (15 agents per request). With a
slow model or provider one of those requests can take minutes.

- Each request waits at most `LLM_REQUEST_TIMEOUT` seconds (default 300, set it in `.env`; raise it for a slow local
  model). After that it is tried again, up to 3 attempts, with a short pause that is longer when the provider
  answers 402 or 429. The preparation page shows which step it is on and why a request is being repeated, and the
  backend log has one line per attempt.
- If the model's answer for an agent is unusable (for example an entry without `agent_id`), only that agent gets the
  rule-based defaults. The rest of the batch is kept.
- An opening post names the kind of agent that should publish it (`poster_type`). The prompt lists the allowed
  types; if the model writes something else, the post goes to an agent whose type or name is close enough
  (`Official` → `GovernmentOfficial`, a name such as `Ministry of Home Affairs`), and otherwise to the most
  influential agents one after another, so that not every post comes from the same account.

---

## 2. Seeds

`POST /api/simulation/start` accepts an optional `seed` (a non-negative integer). Without one the backend picks a
random seed. The seed actually used is stored in `run_state.json` and in the `run` block of the simulation config,
and is returned in the response.

Each platform gets its own random-number stream derived from the seed, so Twitter's schedule does not change
because Reddit consumed a random number.

**What a seed guarantees:** the same seed gives the same *schedule* - who is woken in which round, reaction delays,
the starting follow graph and when scheduled events fire.

You can check this from the logs: every `round_start` record in a run's `actions.jsonl` lists the agents that were
woken that round (`active_agent_ids`), so two runs with the same seed can be compared directly. The logged *actions*
can still differ between such runs: the model chooses what each woken agent does, and an action that fails (liking
a post twice, for example) leaves no record.

**What it does not guarantee:** identical text. The language model's output is not reproducible, so two runs with
the same seed still differ in what the agents write, and those differences feed back into the run. Do not treat a
seed as "replay this run".

---

## 3. Ensembles

An ensemble runs one *prepared* simulation N times. Each run is a clone of the prepared simulation with its own seed
and its own database. Nothing is written to the Zep knowledge graph by an ensemble run: N copies of simulated chatter
would pollute it.

### In the UI

1. In **Step 2**, set *Runs* (1 to 50). `1` is the normal single run and changes nothing. With 2 or more, the
   note under the field tells you how many rounds each run will use. An ensemble always has an explicit round cap
   (the custom rounds if you set them, otherwise the automatic rounds).
2. **Step 3** does not start anything yet. It creates the ensemble and shows a review screen:
   - the **estimated cost** (a conservative upper bound on model calls),
   - the **end-of-run questions** (1 to 3, editable now and locked once the runs start; if you did not supply any,
     one LLM call proposes them from your simulation requirement; if the model's proposal cannot be used the screen
     says so and shows a generic stance question instead),
   - a note on what a seed does and does not do.
   Press *Start N runs* when you are happy.
3. While it runs, each run shows its status, round progress and seed. *Stop all runs* stops the running ones and
   marks the rest stopped.
4. When it finishes you see the **results table** and *Generate report*.
5. **Step 4** shows the same table above the report.

The ensemble's id is added to the page address, so reloading the page continues the same ensemble. An ensemble that
is already running is continued rather than duplicated.

### What each run does at the end: the final poll

After the last round, every agent is asked the outcome questions once, on Twitter when it is enabled and on Reddit
otherwise. The prompt tells the agent to answer as itself, from what it has seen, and to reply with a JSON object only.

| Question type | Answer | What is computed per run |
|---|---|---|
| `probability` | a number from 0 to 100 | mean, median and influence-weighted mean across agents |
| `choice` | one of the given options | the share of agents per option |
| `number` | any number (with a unit) | median and mean |

A `choice` question can carry a `stance_map` (option → supportive / opposing / neutral). It lets the summary count
**stance drift**: how many agents ended on a different side than they started. Answers that cannot be parsed are
excluded from the statistics but counted in the *parse rate*. The poll is tagged `poll` and is never counted as agent
behavior.

Agents that give no answer at all (the request failed, or the model sent an empty text) are asked again after pauses of
5, 15 and 30 seconds, a few at a time and then one at a time, because the usual cause is a provider that throttles
(HTTP 402 "retry after in-flight requests settle", or 429). What stays silent after that shows up as a lower parse rate.

### What you get

`summary.json` (machine-readable) and `summary.md` (readable, opening with the caveat) in
`backend/uploads/ensembles/<ensemble_id>/`:

- per question: the **distribution across runs** (mean, std, min, p10, median, p90, max and how many runs
  contributed), and for probability questions the share of runs whose mean was above 50;
- stance drift (when a stance question exists);
- behavior per run: total actions, posts, comments, likes, reposts, follows, the share of posts by starting stance,
  the agents most often among the top 3 by engagement received, hot-topic mentions (a post or comment counts when
  it contains the phrase, or every word of a multi-word phrase), actions per round;
- a row per run: seed, status, parse rate, actions.

Setup follows, scheduled-event posts and the poll are excluded from the behavior counts, so they do not skew them.

### Reading the numbers honestly

- A *distribution across runs* shows how much the outcome moves when only the random schedule and the model's
  sampling change. A wide spread means the result is fragile; it does not tell you which end is right.
- With few runs the percentiles are rough. The table always shows how many runs contributed.
- Agents answering a poll are role-playing from their persona and what they read in the simulation. Their answers
  are *simulated participants' views*, not forecasts of the real event.
- The persona directive anchors agents on their starting stance. Look at stance drift: if it is close to zero in
  every run, the outcome was largely fixed by the setup and the spread says little. The directive text is a constant
  (`DIRECTIVE_TEXT` in `backend/scripts/sim_behavior.py`) that is easy to soften.

### Cost

The estimate is `runs × platforms × rounds × agents woken per round (upper bound)` plus
`runs × platforms × agents × questions` for the polls. It is intentionally high: real runs wake fewer agents, and the
poll asks each agent once on one platform. A run's cost grows with its rounds, so use fewer rounds first.
The limits are 50 runs, up to 4 runs at once (the UI uses 1 at a time) and a mandatory round cap.

### When things go wrong

- A failed run does not fail the ensemble. With all runs done the status is `completed`; with at least 2 usable runs
  and some failures it is `partial` and the summary uses the runs that worked; with fewer than 2 it is `failed`.
- If the backend restarts while an ensemble is running, runs whose processes are gone are marked failed with the
  error `interrupted`, and runs that had not started are marked stopped. Whatever finished is then aggregated under the
  same rules. Ensembles are not resumed automatically.
- A run's `simulation.log` says why agents stayed silent. With "402" or "429" errors the provider is throttling or out
  of credit: run fewer rounds, or add credit, and try again.
- The run page shows this without opening the log. A status line says what the round is doing ("Round 12: 6 agents
  awake, 2 failed so far"; during the end-of-run questions, how many agents have answered and which retry it is on).
  When most agents' model calls fail, a banner names the cause (no credit, rate limit, rejected key, timeouts, an
  unreachable provider) with the provider's own message. It judges the last 3 rounds, so it disappears once the
  provider recovers. OASIS swallows such errors, so the run scripts record them as `agent_error` events in
  `actions.jsonl`; each `round_end` also carries `failed_count`, and the poll writes `poll_status` events.
- Replicates are hidden from the simulation list and history; `?include_replicates=true` shows them.

### Files

```
backend/uploads/ensembles/<ensemble_id>/
  ensemble.json            definition and status
  outcome_questions.json
  summary.json
  summary.md
backend/uploads/simulations/<base>__<token>__r01 ...   one directory per run (config, profiles, databases, logs,
                                                       final_poll.json)
```

### API

All under `/api/simulation`. Responses are `{"success": bool, "data" | "error"}`.

| Method | Path | Notes |
|---|---|---|
| POST | `/ensemble/create` | `{simulation_id, max_rounds (required), n_replicates=5, concurrency=1, base_seed?, outcome_questions?, llm_temperature?}`. Returns the ensemble, its questions and `cost_estimate`. Nothing runs yet. |
| PUT | `/ensemble/<id>/outcome-questions` | `{outcome_questions: [...]}`; only while the status is `created`. |
| POST | `/ensemble/<id>/start` | Starts the runs in the background. |
| POST | `/ensemble/<id>/stop` | |
| GET | `/ensemble/<id>` | Status, per-run progress (`current_round`/`total_rounds`) and the questions. |
| GET | `/ensemble/<id>/summary` | 404 until the ensemble has been aggregated. |
| GET | `/ensemble/list?simulation_id=` | Ensembles, newest first. |

---

## 4. Reports on an ensemble

`POST /api/report/generate` accepts an optional `ensemble_id` together with the base `simulation_id`. The ensemble
must have finished (`completed` or `partial`) and belong to that simulation. The base simulation itself is never
run when you use an ensemble, so it does not need a finished run.

A report written on an ensemble differs from a normal one in three ways:

- the planning step starts from the ensemble summary;
- the agent gets one more tool, `ensemble_stats`, which returns sections of `summary.json` (overview, one question,
  `polls`, `stance_drift`, `behavior`, `hot_topics`, `replicates`);
- the writing rules tell it to quote **ranges** (p10-p90, min-max), to say plainly when runs disagree, to state the
  caveat above at least once, and to treat `narrative_direction` as a hypothesis to confirm, partly confirm or
  reject.

A report without an ensemble is exactly what it was before. Reports remember their `ensemble_id`; a plain report and
an ensemble report of the same simulation are different documents and one is never reused for the other.

Two limits to know about. The ensemble's runs write nothing to the knowledge graph, so the report's graph searches
describe your source material, not the simulated chatter (the ensemble statistics carry that). And the *interview*
tool needs a live simulation environment, which ensemble runs do not keep.

---

## 5. Where things live

| Piece | Location |
|---|---|
| Behavior logic (pure, unit tested) | `backend/scripts/sim_behavior.py` |
| OASIS glue, final poll | `backend/scripts/sim_runtime.py`, `backend/scripts/sim_poll.py` |
| Ensemble runner | `backend/app/services/ensemble_runner.py` |
| Aggregation (no LLM calls) | `backend/app/services/ensemble_aggregator.py` |
| Outcome questions | `backend/app/services/outcome_questions.py` |
| Scheduled events | `backend/app/services/scheduled_events.py` |
| API | `backend/app/api/ensemble.py`, `backend/app/api/simulation.py`, `backend/app/api/report.py` |
| UI | `Step2EnvSetup.vue`, `ScheduledEventsEditor.vue`, `EnsembleRunPanel.vue`, `EnsembleSummaryTable.vue`, `Step4Report.vue` |

Run the tests with `uv run pytest` from `backend/`.
