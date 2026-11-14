# Free Model Aggregator

One server, two client wires, ten free providers — the entry for
make.fun's `free-model-aggregator-01` ($500).

`main.py` (Python 3.13, standard library only) reads the trial brief,
probes every provider's catalog route, registers its port and catalog
with the env, and serves the env's client as an OpenAI server
(`POST /v1/chat/completions`) and an Anthropic one (`POST /v1/messages`),
in JSON and in server-sent events, exiting 0 when the workload is done.

Routing: for each request, the free models that meet its requirements
are ranked by the quality table's rating (best first), then by how
little their provider has been used (calls plus a penalty per refusal),
then by the brief's provider order. A provider that refuses (402, 429
or 5xx) is skipped for the rest of that request and the request is
retried on the next candidate; a provider that refuses twice in a row
has spent its budget (or hit its outage) and is skipped for the rest of
the trial while any other candidate exists. JSON-mode requests go only
to OpenAI-wire providers, since the other wires cannot carry
`response_format`. Every streamed answer carries its final usage: the
server always asks upstream providers for a usage chunk, and passes the
provider's usage through in the wire the client used.

Everything is deterministic — no wall clock, no unseeded randomness —
so a replay leaves the same state export.

Practice scores with the kit's own env and scorer
(`program/env/server.mjs`, `program/scorer/score.mjs`):

| trial | valid | upstream calls | score (quality / served / wire / failover / catalog) |
|-------|-------|----------------|-------------------------------------------------------|
| 0 | 20/20 | 26 | 99 (99 / 100 / 100 / 100 / 100) |
| 1 | 20/20 | 30 | 99 (99 / 100 / 100 / 100 / 100) |
| 2 | 20/20 | 32 | 99 (99 / 100 / 100 / 100 / 100) |

`python3 main.py --list` prints the catalog document and exits 0.
