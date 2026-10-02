# model-drift-watch for Hermes Agent

Find out when the model you use gets quietly worse.

Same model name, weaker answers: a smaller or quantized model behind the name, less reasoning
effort, a routing change. People notice it as "it feels dumber today" and then argue about it.
This plugin replaces the argument with a measurement. Every day it asks your model the same
small set of questions, grades the answers exactly (no model judges another model), and tests the
result against **the same model's own baseline**. You get a message only when a
drop passes that test (and a same-day second run), when a different model is answering, or when
the measurement itself failed.

It runs inside Hermes: Hermes cron schedules it, Hermes keeps the API keys, and results are
delivered by Hermes cron to the chat you choose (Telegram, Slack, Discord, and the other Hermes
platforms).

## What has been tested

It has not been tried against a real LLM API yet. What has been checked is a local mock model
(`tests/mock_openai_server.py`, an OpenAI-compatible server that answers from the question set and
can be switched to a degraded, rate-limited or different model) and real Hermes cron: in Hermes
0.21.5 the scheduled job ran the tool through Hermes's tool bridge and `ctx.llm` against that
mock. Those scheduled runs used `--deliver local`, so delivery to a real Telegram, Slack or Discord
chat has not been tried either. Answer lengths, rate limits and error formats of real providers
may differ from the mock.

## Install (3 commands)

Needs Hermes 0.21.5 or later.

```bash
hermes plugins install TakeshiTGAL/hermes-plugin-model-drift --enable   # once listed in the catalog: hermes plugins install model-drift-watch --enable
hermes model-drift-watch schedule --deliver telegram     # daily at 09:00; slack, discord, ... work too
hermes gateway restart                                   # the gateway runs scheduled jobs (`hermes gateway` if it is not running)
```

To try a local checkout instead: `hermes plugins install file:///path/to/hermes-plugin-model-drift --enable`.

That is all. It watches the main model in your Hermes config. The first 5 daily checks build the
baseline (you get one message when it is ready); after that you hear from it only when something
changed. Before anything is spent you can see the cost:

```bash
hermes model-drift-watch estimate      # free: tokens and dollars per check, and whether it fits your cap
hermes model-drift-watch run           # optional: run a check now
hermes model-drift-watch status        # free: last result and the schedule
```

The same three actions exist as the slash command `/model-drift-watch` in any chat, and as the tool
`model_drift`, so you can simply ask Hermes "has my model got worse?".

Tip: the scheduled job reaches the tool through one short Hermes agent turn. If your main
provider is down, that turn fails too and you get Hermes's own cron failure notice instead of a
model-drift-watch message. Running that turn on a model from another provider avoids it and is cheaper:
`hermes model-drift-watch schedule --deliver telegram --agent-model <model> --agent-provider <provider>`.
The model being tested stays your main model.

## What the messages look like

Outputs of scheduled runs in Hermes 0.21.5 (Hermes cron, the tool bridge, `--deliver local`)
against the local mock model, switched into a degraded, rate-limited or different model:

```text
[model-drift-watch] DRIFT: custom:mock-model is answering worse than its own baseline.
Accuracy 68% now vs 89% baseline (-21 points, 95% CI -37 to -6), exact test p=0.0011.
A second run right after agreed: 74% (-15 points, p=0.01; both runs together p=0.0001).
Significant in: reasoning -30 points.
Served model: mock-model-1 (baseline: mock-model-1).
Next: if you changed this model on purpose, run `hermes model-drift-watch reset-baseline`. Otherwise compare with another provider of the same model; `hermes model-drift-watch status` has the numbers.
Spent 10,252 tokens in 2 runs (estimate 5,857 per run), cost unknown.
```

```text
[model-drift-watch] Could not measure custom:mock-model this time. This is not a verdict on the model.
34 of 34 answers could not be used (rate_limited 34).
Your provider is rate-limiting. Lower concurrency (hermes config set plugins.entries.model-drift-watch.settings.concurrency 2) or schedule the check at a quieter hour.
Spent 0 tokens (estimate 5,857 per run), cost unknown.
```

```text
[model-drift-watch] MODEL CHANGED: answers for custom:mock-model came from "mock-model-2", but the baseline was answered by "mock-model-1".
Usually Hermes fell back to another model after the main provider failed, or the provider changed what runs behind the name.
On the same questions: 79% now vs 89% baseline.
Next: if this is the model you want, run `hermes model-drift-watch reset-baseline`; if not, look for fallback lines in `hermes logs`.
Spent 5,126 tokens (estimate 5,857 per run), cost unknown.
```

| Status | Message? | Meaning |
|---|---|---|
| `collecting` | no | Building the baseline (k of 5 runs). |
| `baseline_ready` | yes, once | Testing starts with the next run. |
| `stable` | no | Matches the baseline. |
| `drift` | **yes** | Significantly and materially worse, confirmed by a second run the same day. |
| `token_shift` | **yes** | Answers became much shorter or longer (confirmed) while accuracy did not move significantly; often a different reasoning effort behind the same name. |
| `model_changed` | **yes** | The provider reports a different model id than in the baseline (often Hermes falling back after a provider error). |
| `measurement_failed` | **yes** | Rate limits, outages, empty or rerouted answers above 20%. Never reported as drift. |
| `suspect` | yes | One run looked significantly worse but the confirmation run could not happen (cap, time) or be measured. |
| `unconfirmed`, `improved` | no | Shown in `status`; set `notify: always` to get every result. |

## How it decides

**Questions.** 34 short questions in three areas by default: reasoning (13), instruction following
(10) and Japanese (11). Ten more code questions run only if you turn on `code_execution` (see
"Code questions" below). Most are easy for a strong model on purpose (a broken model fails them),
and some are the kind strong models still get wrong now and then: long multiplication, counting
letters or kana, sorting letters, reversing strings. Every answer is graded by a fixed rule:
exact text, a number, regular expressions, a JSON value, or hidden unit tests run against the
code. The test suite recomputes every answer key that can be computed, and every bundled question
has a reference answer that must pass its own grader.

**Test.** Each question is compared only with itself. Under "nothing changed", the right answers
of a question are equally likely to fall in any of its runs, so given how many of a question's
answers were right in total, the number that land in today's run is hypergeometric. Today's total
is the sum over questions; its exact distribution is a convolution, which gives an exact p-value
with no approximations and no random numbers. Question difficulty drops out, and questions the
model always gets right (or always wrong) weigh nothing until they change.

**Areas.** The whole set gets 60% of the significance budget and the areas share the other 40%
(Holm), so a collapse in one area cannot hide behind a good average, while the accuracy alert's
false-alarm rate stays at or below `alpha` (1%) on average over baselines, if runs are
independent; a particular frozen baseline can be luckier or unluckier (Table 2). (While the
baseline is still growing it only takes in runs that ended clean, which tilts it slightly; Table 2
below measures that effect on false alarms. A drop that starts while the baseline is still
growing can join it, so such a drop is caught less often than Table 1 shows until the baseline
freezes.)

**Confirmation.** A first run that looks worse at the 5% level triggers a second run right away.
An alert needs both runs together to pass the 1% test, the second run on its own to show the drop
(p < 0.05), and a drop of at least 5 accuracy points. Answer length (output tokens, which include
hidden reasoning tokens) is tested the same way with an exact sign test and its own 1% bound,
and needs a change of at least 25%.

**Baseline.** Testing starts after 5 successful runs that are at least 12 hours apart (a burst of
manual runs cannot make a too-narrow baseline). After that, runs of checks that ended clean keep
joining the baseline up to 15, so one lucky early run cannot set the bar for good; then it is
frozen, so a slow decline is not absorbed. Change models on purpose? `hermes model-drift-watch
reset-baseline`. Edit or add a question (or change `max_tokens_scale`) and only the questions
concerned collect their own 5 answers again, also after the freeze; the rest keep being tested.
The "baseline ready" message says how many questions your model answered right only some of the
time: those carry most of the information, and if there are very few, add harder questions of
your own.

### False alarms and detection, measured

From `python scripts/simulate_error_rates.py --trials 20000 --users 200 --days 300 --items 34`
(34 questions in three areas, default settings). "Strong" gives every question an 80-100% chance of a right answer, "mixed"
50-100%. Table 1 is the average over users with the smallest (5-run) baseline:

| Model profile | Questions newly wrong | Accuracy alert, one run only (`confirm: false`) | Days with a confirmation run | **Accuracy alert (default)** |
|---|---|---|---|---|
| strong | 0% (nothing changed) | 0.36% | 2.18% | **0.16%** |
| strong | 5% | 2.45% | 8.75% | 2.00% |
| strong | 10% | 4.60% | 16.20% | 5.85% |
| strong | 20% | 42.75% | 75.50% | **68.00%** |
| mixed | 0% (nothing changed) | 0.47% | 2.54% | **0.22%** |
| mixed | 10% | 2.20% | 10.05% | 2.95% |
| mixed | 20% | 19.60% | 45.60% | 30.30% |

A frozen baseline makes some users luckier than others, so Table 2 follows 200 users for 300 days
each with nothing changed (baseline growing to 15 as described):

| Model profile | Median user | 95th percentile user | Worst user |
|---|---|---|---|
| strong | 0.00% | 0.67% | 1.67% |
| mixed | 0.00% | 1.00% | 1.67% |

What this means in practice:

- **False alarms:** about 0.2% of daily checks on average (roughly one in a year and a half), up
  to about 1-2% of days for the unluckiest baselines. The answer-length alert added 0.00% in the
  simulation with ordinary length variation and 0.03-0.07% when answer lengths vary wildly
  (`--sigma 0.8`); volatile lengths also raise confirmation runs to about 4% of days.
- **A model that gets 20% of the questions newly wrong** (a strong model losing ~18 points) is
  caught on 68% of days, about 90% within two daily checks if the two days are independent.
- **Small drops are out of reach** for 34 questions: losing 5% of the questions is caught on only
  2% of days. For finer detection, raise `samples_per_item` to 2 (double cost) or add harder
  questions of your own; questions your model gets right only sometimes carry the most
  information. If your model gets every question right every time, a degradation that does not
  make it miss any of them cannot be seen.
- The simulation assumes answers are independent from run to run. A provider whose quality
  swings from day to day will produce more confirmation runs and more alerts: the same-day
  confirmation run shares that day's conditions, so it cannot tell a bad day from a lasting drop.

## What it costs

API calls go to your provider and are paid by you. For the default set the plugin estimates
about 9,200 tokens for the first run (34 short questions, a first-run guess for answer length)
and then uses the model's own measured answer lengths. Reasoning models that think before
answering spend more; the estimate adapts after one run.

Rough expected cost of a first check (input/output price per million tokens): $0.01 at $0.25/$1,
$0.10 at $3/$15, $0.16 at $5/$25. A day that needs a confirmation run costs twice that. Everything
here is per watched model: watching two models costs twice as much.

Guards:

- `hermes model-drift-watch estimate` prints the expected and worst-case tokens and dollars before
  anything is sent, and warns when two runs (a confirmation day) would not fit your cap.
- A check whose expected cost is over `max_tokens_per_check` (120,000) or `max_usd_per_check`
  ($0.50) does not start. A confirmation run only starts if a whole run still fits.
- During a check, a request is sent only while its longest possible answer (plus 1.3x the
  estimated input) still fits under the cap. A check stopped by the cap is a failed measurement,
  not drift.
- What the plugin's own accounting cannot see: input a provider bills for a request that then
  failed (timeouts and 5xx are counted at the estimated input, other failures as zero), retries
  and fallbacks inside Hermes, and the scheduled agent turn below.
- Prices come from Hermes's own pricing code (its bundled table, or, depending on the provider,
  OpenRouter, models.dev or the provider's own `/models` endpoint); set `price_input_per_mtok` and `price_output_per_mtok` if your
  provider is not covered (otherwise only the token cap applies).
- The scheduled agent turn that calls the tool costs about 5,200-5,800 input tokens in a fresh
  Hermes 0.21.5 (two requests, ~2,400 and ~2,700-3,400 tokens, estimated from their size by the
  mock server), more if your system prompt is large. It is not counted against the cap. Use
  `--agent-model` (see the tip above) to put it on a cheap model.

## Time limits

Hermes stops a tool call after `timeouts.tools.sequential_call` (420 s by default). One check
must finish before that, for every watched model together: `check_deadline` (default 300 s, and
never more than your `sequential_call` minus 30 s) is one shared budget for the whole call, each request's timeout shrinks to the time left, and no new
question or confirmation run starts that could not finish. Questions not asked in time count as
unusable answers, which ends as a failed measurement, never as drift. With the default 4 parallel
requests a run takes about 9 times as long as one answer, so a model that needs 20 s per answer
uses most of the budget; raise `concurrency` for slow models if your provider allows it.

## Settings

All settings live under `plugins.entries.model-drift-watch.settings` and show up in the Desktop
Plugins tab. From the command line:

```bash
hermes config set plugins.entries.model-drift-watch.settings.max_usd_per_check 1.0
```

| Setting | Default | |
|---|---|---|
| `max_tokens_per_check` / `max_usd_per_check` | 120000 / 0.5 | Hard caps per check, confirmation included |
| `price_input_per_mtok` / `price_output_per_mtok` | 0 (use Hermes pricing) | USD per million tokens |
| `baseline_runs` / `baseline_max_runs` | 5 / 15 | Runs before testing starts / largest baseline |
| `baseline_min_gap_hours` | 12 | Spacing between baseline runs (0 = off) |
| `samples_per_item` | 1 | 2 doubles cost and improves sensitivity |
| `alpha` / `min_drop_points` | 0.01 / 5 | Significance bound and smallest drop worth an alert |
| `confirm` | true | Same-day confirmation run before alerting |
| `max_error_rate` | 0.2 | Above this share of unusable answers the run is a failed measurement |
| `concurrency` / `request_timeout` / `check_deadline` | 4 / 90 s / 300 s | See "Time limits" |
| `max_tokens_scale` | 1.0 | Multiplies every question's answer limit; raise it (e.g. 4) for reasoning models that return empty answers. Starts a new baseline |
| `temperature` / `send_temperature` | 0 / true | Turn off sending for models that reject it |
| `code_execution` | false | Turn on to also run the 10 code questions (see "Code questions") |
| `notify` | problems | `always` sends every result |
| `categories` | [] (all) | e.g. `["reasoning", "code"]` |
| `extra_targets` | [] | More models to watch, see below |

## Code questions

Ten bundled questions ask for a Python function and grade it by running hidden tests. They are
off by default (`code_execution: false`), because the code is written by your model and runs on
this machine, also in scheduled runs where nobody approves it. Hermes refuses its own
`execute_code` in cron unless `approvals.cron_mode` is `approve`; turn this on only if you accept
the same:

```bash
hermes config set plugins.entries.model-drift-watch.settings.code_execution true
```

Turning it on adds the ten questions, which collect their own 5-run baseline before they are
tested (the other questions keep being tested).

## Your own questions

Put `.json`, `.jsonl` or `.yaml` files in the folder printed by
`hermes model-drift-watch validate-suite` (`<hermes home>/plugin-data/model-drift-watch/suites/`):

```yaml
items:
  - id: refund-policy-1
    category: support
    prompt: "A customer bought a plan on 3 March and asks for a refund on 20 March. Our policy allows refunds within 14 days. Can we refund? Answer yes or no."
    grader: {type: exact, answers: ["no"]}
    reference: "ANSWER: no"
  - id: sql-1
    category: support
    extract: full
    prompt: "Write only a SQL query that counts rows in table orders. No code fences."
    grader: {type: regex, patterns: ["(?i)^select\\s+count\\(\\*\\)\\s+from\\s+orders;?$"]}
```

Graders: `exact` (`answers`, optional `case_sensitive`, `strict`), `number` (`answer`,
`tolerance`), `regex` (`patterns` all must match, `forbid` none may), `json` (`equals`,
`unordered`), `python` (`entry`, `tests`; needs `extract: code`). `extract` is `answer_line`
(default; the model is asked to end with `ANSWER: ...`), `full`, or `code`. Then run
`hermes model-drift-watch validate-suite`: it reports every problem with the fix, and checks that each
`reference` passes its own grader.

## Watching more than one model

Your main model is watched with no setup. `hermes model-drift-watch targets` lists every model in your
config (fallbacks, auxiliary tasks, delegation). To watch another one, add it to `extra_targets`
(`"provider:model"`). Hermes lets a plugin choose a model other than your active one only after
you allow it, and the `targets` command prints the exact `config.yaml` block
(`plugins.entries.model-drift-watch.llm.allow_model_override` with an `allowed_models` list).

## Disclosure

What you should know before installing:

- **Cost:** every check asks your model 34 questions (44 with `code_execution` on) through your own provider, on your bill
  (see "What it costs"). The plugin stops itself at `max_tokens_per_check` / `max_usd_per_check`.
  The scheduled agent turn that calls the tool is billed too and is not under that cap.
- **Network:** the plugin opens no connections itself. Its LLM requests go to your own provider
  through Hermes (`ctx.llm`), and the price lookup goes through Hermes's own pricing code, which
  may fetch prices from OpenRouter, models.dev or the provider's `/models` endpoint. Hermes holds
  the credentials; the plugin never sees a key. No telemetry, no update checks.
- **Code execution:** off by default. When you turn it on, each code answer is graded in a child
  process: the same Python started in isolated mode (`-I -S -B`) in an empty temporary folder,
  with an environment that holds no secrets, CPU/memory/file-size limits where the OS supports
  them (memory is not limited on macOS) and a 10 s timeout. Inside it an audit hook refuses
  ordinary file writes, network, further processes and ctypes, and reads outside the Python
  installation and the temporary folder. Code written to get around it can (for example by
  replacing built-ins), so this is not a sandbox for hostile code. See "Code questions".
- **Reads:** `config.yaml` (model names and this plugin's settings) and the cron job list (to show
  the schedule).
- **Writes:** run history under `<hermes home>/plugin-data/model-drift-watch/` (append-only JSON lines:
  right/wrong per question, token counts, the served model id, error codes with a short error text,
  and the report; not the answer text), and one cron job named `model-drift-watch` when you run
  `schedule`.
- **Messages:** the plugin sends nothing itself. Hermes cron delivers the job's reply to the
  `--deliver` target you chose (default `local`: saved under `<hermes home>/cron/output/` only).
  On quiet days the reply is `[SILENT]` and nothing is delivered.
- **Removing it:** run `hermes model-drift-watch unschedule` before `hermes plugins remove
  model-drift-watch`. Removing or disabling the plugin does not remove the cron job, which would
  keep running one agent turn a day. If that happened, `hermes cron list` shows the job and
  `hermes cron remove <id>` deletes it. The history folder stays until you delete it.
- **Hermes internals used:** scheduling calls `cron.jobs.create_job` / `remove_job` /
  `list_jobs` (the functions behind `hermes cron`), because plugins have no public cron
  registration yet; it restricts the job to this plugin's toolset. Nothing in Hermes is patched
  or replaced.

## Compared with livenerf

[livenerf](https://github.com/ninjahawk/livenerf) is a public, pre-registered 30-day study of
Claude Opus 5.5 as served through Claude Code. model-drift-watch takes ideas described in its
README (a model's own baseline, exact grading, item-paired statistics, output-token count as an
early signal) and applies them to a monitor for the models you run in Hermes. The two do different
jobs. The livenerf column is based on its README as read on 2026-10-02:

| | livenerf | model-drift-watch |
|---|---|---|
| Purpose | One public 30-day series: Claude Opus 5.5 via Claude Code on a subscription | Ongoing monitor for the models you use in Hermes |
| Setup | Clone, `uv sync`, pin the Claude Code CLI, then calibrate, design and validate a panel | Install the plugin and schedule it inside Hermes |
| Models | Claude through `claude -p` | Any Hermes provider, read from your config; extra models opt-in |
| Schedule | crontab / Windows Task Scheduler | Hermes cron (`hermes cron list`) |
| Results | A running 10-day results table and `livenerf.analysis` | Message to your chat platform when something changed |
| Questions | 78 GPQA Diamond, MMLU-Pro and competition-math questions the model gets right only sometimes, chosen by calibration | 34 short items incl. instruction following and Japanese (44 with code turned on); add your own |
| Grading | Exact match, hidden tests for code | Exact, number, regex, JSON, hidden tests in a locked-down process |
| Statistics | Per-item paired differences, clustered SEs, 99% interval in two consecutive 10-day windows | Exact per-question conditional test, area split with Holm, same-day confirmation |
| Time to a verdict | ~30 days (10-day baseline + two 10-day windows) | After 5 baseline runs, then every check |
| Sensitivity | ~7.5 accuracy points per 10-day window (its README) | ~18 points caught on 68% of days; small drops out of scope (see tables) |
| Provider failures | Samples touched by the safety classifier are rejected and counted | Rate limits, outages, fallbacks, empty answers: "could not measure", never drift |
| Serving changes | Control arm with a second model | Served model id checked against the baseline; answer-length shift reported |
| Cost control | Share of the plan's weekly meter | Tokens and dollars estimated before running; caps enforced by the plugin |
| License | MIT, stated in its README (no LICENSE file in the repository) | MIT |

Prior art and credits are listed in [NOTICE](NOTICE). No code from any of them is included.

## Development

```bash
python -m pytest -q                                   # 85 unit and end-to-end tests, no network, no API key
python scripts/simulate_error_rates.py                # the false-alarm / detection tables
python tests/mock_openai_server.py --port 18765       # OpenAI-compatible mock for end-to-end runs
```

The mock serves both the probe questions and the scheduled agent turn, and its `/control`
endpoint switches it to a degraded model, a rate-limited provider, an outage, a different served
model or shorter answers, so the whole path (Hermes cron, the tool bridge, `ctx.llm`, grading,
the decision, the job's reply) can be exercised without spending anything. Three checks to run
this way, after five runs have built the baseline (`hermes cron run <id>` runs the job now;
set `baseline_min_gap_hours` to 0 first):

- 20% of the questions answered wrong (`{"degrade": 0.2}`): `DRIFT`, confirmed by a second run.
- Nothing changed, 10 runs: no alert (`[SILENT]` every time).
- Every request rate-limited (`{"rate_limit": 1.0}`): "Could not measure ... This is not a verdict
  on the model", not drift.

## 日本語

model-drift-watch は、使っているモデルが同じ名前のまま弱くなっていないかを毎日確かめる Hermes Agent
のプラグインです。決まった34問（推論・指示どおりの出力・日本語。コードの10問は自分で有効にしたときだけ）を毎日同じ形で出し、
答えを機械的に採点して、そのモデル自身の基準と比べます。偶然のぶれでは鳴らず、本当に下がった
ときだけ Telegram や Slack などに知らせます。

実際の LLM の API ではまだ試していません。確かめたのは、ローカルの模擬モデルと、Hermes 0.21.5 の
定期実行（cron）からその模擬モデルに対して動かすところまでです。定期実行の結果は手元に保存する設定
（`--deliver local`）で確かめたので、Telegram などへ実際に届くかも未確認です。

導入は3行です（Hermes 0.21.5 以降）。

```bash
hermes plugins install TakeshiTGAL/hermes-plugin-model-drift --enable
hermes model-drift-watch schedule --deliver telegram
hermes gateway restart
```

外すときは、先に `hermes model-drift-watch unschedule` で定期実行を消してから
`hermes plugins remove model-drift-watch` を実行してください。プラグインを外しても定期実行は残ります。

- **判定の仕方:** 問題ごとに「基準の回と今日の回のどちらに正解が入ったか」を数える正確な検定を
  使います。全体と分野別を合わせて、正答率の誤報は、基準の取り方を平均して1%以下です（回ごとの結果が
  独立という前提。基準の運しだいで、人によっては1〜2%。表2）。
  怪しい日はその場でもう1回解かせ、2回分を合わせて確認できたときだけ知らせます。
  シミュレーションでは、何も変わっていないモデルで鳴るのは平均0.2%の日（基準の運が悪い人でも
  1〜2%）、2割の問題を新たに間違えるようになったモデルは68%の日で、2日以内なら約90%で
  見つかります。数点の小さな低下は34問では見分けられません。
- **基準:** 12時間以上あけた最初の5回で判定を始め、その後も問題のなかった回を15回まで足して
  から固定します。ゆっくりした低下を基準が吸い込まないためです。ただし、基準が固定される前に
  始まった低下は基準に入ることがあり、そのあいだは上の68%より見つけにくくなります。
- **費用:** API 代は利用者の負担です。実行前に `hermes model-drift-watch estimate` で予想トークン数と
  金額を表示し、上限（既定 12万トークン・0.5ドル）を超える見込みなら始めません。実行中も、
  上限を超えうる問い合わせは送りません。
- **測れなかった日:** 429（回数制限）、通信断、別モデルへの切り替わり、空の返答が2割を超えた日は
  「測定失敗」と知らせ、劣化とは言いません。
- **コードの問題:** 既定では無効です。有効にすると、モデルが書いた Python をこの機械で動かします
  （人が承認しない定期実行の中でも）。Hermes 自身も、`approvals.cron_mode` が `approve` でない限り、
  定期実行の中で execute_code を動かしません。同じことを受け入れる場合だけ有効にしてください。
- **自分の問題:** `hermes model-drift-watch validate-suite` が示すフォルダに JSON か YAML を置けば
  追加できます。

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
