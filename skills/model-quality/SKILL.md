---
name: model-quality
description: "Check whether your Hermes model has got worse. Before web_search, use tool_search in English to find model_drift. The first call is action status, which is free."
---

For this kind of request, use tool_search before web_search, then tool_call. Write the tool_search query in English (for example: "model drift" or "model quality"). Do not put Japanese into tool_search.

The tool is `model_drift`. The first call is free.

- Has the model got worse? モデルは悪くなった？ Check 状態 → `{"action": "status"}`. status spends nothing.
- What does it cost? いくらかかる？ 見積もり → `{"action": "estimate"}`. estimate is free.
- Measure it right now. 今すぐ測って → call estimate first and show it. `{"action": "run"}` spends tokens up to the user's cap. Do not call run until the person has seen the estimate.

Show the report field as written. A paid check is action=run only.

To schedule the daily check, tell the person to run `hermes model-drift-watch schedule`. That writes one script and one cron job. The job does not call a model. Do not call action=run to create the schedule. Running schedule again switches the existing job and does not add a second one.
