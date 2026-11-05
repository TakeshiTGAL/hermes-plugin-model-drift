# Changelog

## 0.2.1

`status`, `estimate`, and `run` no longer treat a blank model as `auto:default`. Hermes stores `model: ""` until you pick one. The reply says to run `hermes model`, then `status` again, and the check does not call a model.

A rejected request mentions `send_temperature false` only when the provider's error mentions temperature. When the provider says the model does not exist, or returns 404, the next step is to check the model name.

## 0.2.0

schedule writes `HERMES_HOME/scripts/model-drift-watch.py` and creates one cron job with `hermes cron create --no-agent --script`. When `HERMES_HOME` is unset, that directory is the Hermes default, from `hermes_constants.get_hermes_home()`. The job runs that script. The script runs `hermes model-drift-watch run --json` and prints the message. Empty stdout sends nothing. The questions still go through `ctx.llm` and still count against the cap. There is no agent turn on a job this version creates.

An existing job named `model-drift-watch` is switched in place with `hermes cron edit --script --no-agent`. The id stays the same. schedule does not remove it first and does not create a second job. A job made by 0.1.x keeps its model turn until that next schedule. If the job list cannot be read, nothing is changed.

`--agent-model` and `--agent-provider` are gone, because there is no agent turn to put on another model. unschedule removes that job and that script and keeps the history. The plugin does not import the cron job store. `requires_hermes` is `>=0.21.4`. The slash command does its work on a worker thread, so a check does not stop the other gateway chats.

## 0.1.2

The first check is `hermes model-drift-watch status`. It does not call the model and does not spend. The old three-command schedule is still there, after that.

`action` accepts these words, 状態 and 確認, as status, and 見積もり as estimate. Both are free. 実行, 計測, and 測定 do not start a paid check: the reply says to call action=estimate first, and nothing is spent. `action=run` is unchanged and still spends up to the cap.

The skill names those three Japanese requests and tells tool_search to use English.

## 0.1.1

Adds the `model-quality` skill. It tells the model to find `model_drift` with tool_search before web_search. It does not set requires_toolsets, so it stays in the model's skill list when Tool Search is left at its default, which hides this plugin's tools.

Fullwidth digits and a fullwidth colon in target are accepted. action in any letter case, and a missing action (read as status), were already accepted; action is no longer marked required in the schema. Extra arguments, which used to be ignored, are refused, and an unknown action such as 10.0 is refused as before. Each refusal gives a fix and an example, does not copy the caller's value, and adds `ok: false` and `error_type`.

The first sentence Tool Search shows now uses the words a person would ask with. A sentence that starts by describing the mechanism is not found, and the model falls through to the built-in web_search.

The README's Disclosure section now says who can call the tool and how to turn it off per surface, and that the skill adds one line to the skills list of every session, including surfaces where the tool is turned off.
