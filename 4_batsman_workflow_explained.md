# How the Batsman Parallel Workflow Works

This explains [4_batsman_workflow.ipynb](4_batsman_workflow.ipynb) — a graph
that fans out to three nodes running **in parallel**, then fans back in to
one node before finishing. It also covers a real bug the workflow hit and
why the fix was needed, since that's the part that trips people up the first
time they write a parallel LangGraph branch.

## 1. The state

```python
class BatsmanState(TypedDict):
  runs: int
  balls: int
  fours: int
  sixes: int
  sr: float
  bpb: float
  boundary_percent: float
```

`runs`, `balls`, `fours`, `sixes` are given up front. `sr` (strike rate),
`bpb` (balls per boundary), and `boundary_percent` are each computed by a
separate node.

## 2. The shape of the graph

```
        ┌──────────────┐
        │     START     │
        └───────┬───────┘
     ┌───────────┼───────────┐
     ▼            ▼            ▼
calculate_sr  calculate_bpb  calculate_boundary_percent
     │            │            │
     └───────────┼───────────┘
                  ▼
              summary
                  │
                  ▼
                 END
```

`START` has three outgoing edges, so LangGraph runs `calculate_sr`,
`calculate_bpb`, and `calculate_boundary_percent` **in the same step** —
that's the "parallel" part. All three then edge into `summary`, so
`summary` only runs once, after all three finish (the "fan-in").

## 3. The bug: returning the whole state from a parallel branch

The first version of each node looked like this:

```python
def calculate_bpb(state: BatsmanState):
  state['bpb'] = state['balls'] / (state['fours'] + state['sixes'])
  return state          # returns the ENTIRE state dict
```

Each of the three branch functions did the same thing: mutate one field on
the full `state` dict, then return the whole dict — including `runs`,
`balls`, `fours`, and `sixes`, which none of them actually changed.

That matters because of how LangGraph merges the outputs of nodes that ran
in the same step. By default, every field in the state (`runs`, `balls`,
...) is backed by a `LastValue` channel, and a `LastValue` channel accepts
**at most one write per step**. Since all three nodes ran in the same step
and all three returned a dict containing `runs`, `balls`, `fours`, and
`sixes`, LangGraph saw three simultaneous writers to the same channels and
raised:

```
InvalidUpdateError: At key 'runs': Can receive only one value per step.
Use an Annotated key to handle multiple values.
```

(If you don't see that exact error, you may instead see silently wrong
numbers like `bpb: 0.0` — depends on exactly what state each branch happened
to be holding when it returned. Either way, the root cause is the same:
three nodes claiming to "own" fields they didn't actually compute.)

## 4. The fix: return only what you changed

```python
def calculate_sr(state: BatsmanState):
  return {'sr': (state['runs'] / state['balls']) * 100}

def calculate_bpb(state: BatsmanState):
  return {'bpb': state['balls'] / (state['fours'] + state['sixes'])}

def calculate_boundary_percent(state: BatsmanState):
  return {'boundary_percent': ((state['fours'] + state['sixes']) / state['balls']) * 100}
```

Each node still *reads* whatever fields it needs from `state`, but now
*returns* a small dict containing only the one field it's responsible for.
LangGraph treats a node's return value as a **partial update** — "here's
what changed" — not a replacement for the whole state. Now each of the
three parallel branches writes to a different channel (`sr`, `bpb`,
`boundary_percent`), so there's no longer any conflict: one writer per
channel per step.

`summary` is unaffected by this — it runs by itself, after the fan-in, so
it's fine for it to read (and return) the full state; it's the only node
touching those channels at that point.

**Rule of thumb:** when multiple nodes can run in the same step (parallel
branches from a fan-out), each one should return only the keys it actually
computed. Only return/mutate the full state dict when a node is the sole
writer in its step.

## 5. Walking through a run

```python
initial_state = {'runs': 75, 'balls': 42, 'fours': 7, 'sixes': 3}
final_state = workflow.invoke(initial_state)
```

1. `START` fires `calculate_sr`, `calculate_bpb`, and `calculate_boundary_percent`
   together, each reading the same input (`runs=75, balls=42, fours=7, sixes=3`).
   - `calculate_sr` → `{'sr': 75/42*100}` = `{'sr': 178.57}`
   - `calculate_bpb` → `{'bpb': 42/(7+3)}` = `{'bpb': 4.2}`
   - `calculate_boundary_percent` → `{'boundary_percent': (7+3)/42*100}` = `{'boundary_percent': 23.81}`
2. LangGraph merges these three partial updates into one state, since each
   touched a different key: `{runs, balls, fours, sixes, sr, bpb, boundary_percent}` — all seven fields now set.
3. `summary` runs once with the fully merged state and prints all seven values.
4. The graph reaches `END` and `workflow.invoke(...)` returns the final
   merged state as `final_state`.
