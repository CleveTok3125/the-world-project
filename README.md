# The World Project

A small simulation of three villages that feed themselves, keep a stock of
commodities, and bargain with each other to stay fed.

Time only moves when you tell it to, one day at a time.

## Running

Needs `numpy`, `scikit-learn` and `textual`, which `pip install -e ".[dev]"` pulls
in from `pyproject.toml`.

```bash
python run.py                                # step through the days yourself
python run.py --days 30                      # let thirty days run on their own
python run.py --days 30 --seed 7 --noise 0.2
python run.py --days 30 --log run.txt        # write the whole run to a file
```

`python run.py` always opens a full screen interface built with `textual`, and it
opens on a chat rather than on the tables.

- The tables sit in two columns above: the day that just ran on the left, the
  settings and the run summary on the right.
- The chat takes up the lower third of the window and follows itself down to the
  newest turn unless you scroll up to read back.
- A marker shows in it while the day is being simulated.
- You give your subordinate a change in the box and they answer as
  `[The Subordinate]`. The narrator writes the day into the same chat as
  `[The Narrator]`, so an instruction, its answer and the story all sit together.
- Each turn carries its own background, and the marker above the box says whichever
  of the two is working.
- `n` or `space` moves to the next day, `q` leaves.
- Typing takes the keys while the chat box has focus, so `escape` hands them back.

Narrating a day can take a while when a language model is behind it, so the day is
simulated on a worker thread and the panel shows a spinner with a status line while
it works. Pressing `n` again while a day is still running does nothing rather than
piling up work, and a narrator that fails is reported in place without stopping the
world.

With `--days` the run needs no keys: it simulates that many days in a row and
closes itself, which is the shape to use when you want the answer in a file.

| Flag      | Meaning                                                   |
| --------- | --------------------------------------------------------- |
| `--days`  | Number of days to simulate. Zero means wait for the keys. |
| `--seed`  | Random seed. One is drawn when left out.                  |
| `--noise` | Relative amplitude of the day to day production waver.   |
| `--log`   | Write the run to this file as it goes.                    |

A day has room for `time_per_day` barters and no more, so not every pair of villages
manages to meet and which ones do changes from day to day. A pair where somebody
runs short is served first, so a village in need always gets its turn. That limit is
the only one: the villages keep no record of what they traded before, so a commodity
may travel from one village to another and straight back the next day. What decides
is whether the day still has room for one more.

A run draws its own seed and shows it, so two runs rarely look the same. Passing
`--seed` back replays that run exactly.

`--log` takes a path. Each day is appended as soon as it is simulated, so a long
run can be watched while it goes, and the run summary is written when the
interface closes. A name ending in `.jsonl` gives one JSON object per day instead
of prose.

The run summary, kept in the right column and repeated at the end of the log, says
how many days went in balance, which day was the worst, and the verdict on the
final one.

## The world

Three villages, each specialised in one commodity and each short of the other
two. They produce exactly what the three of them consume, so nothing enters or
leaves the world and only the barters move goods around.

The stock table shows each village's holdings as a bar scaled to the largest
holding on screen, with a `Total` column and three rows underneath. Holdings carry
no colour, which leaves colour free to mean the same thing on every total row:

| Row         | Colour    | Meaning                                                    |
| ----------- | --------- | ---------------------------------------------------------- |
| `Produced`  | green     | What the three villages made of each commodity that day     |
| `Consumed`  | amber     | What they ate of each commodity that day                    |
| `Difference`| blue      | `Produced` less `Consumed`, so it reads negative when the world is living off its stores |

The colours come from the active Textual theme, so they follow it rather than
being fixed.

| Village   | Specialises in |
| --------- | -------------- |
| Farmers   | Agriculture    |
| Miners    | Minerals       |
| Artisans  | Handicrafts    |

### One simulated day

1. Every village produces, then consumes what it can.
2. Every brain learns from the state it just observed.
3. Every pair of villages bargains once, and exchanges goods if they agree.

## The barter

`Village.trade` is the only way goods ever move. One exchange trades a single
commodity for a **different** commodity, and both sides state how many units
they give and how many they want.

The exchange is atomic. When any precondition fails, nothing moves: the two
commodities must differ, both quantities must be positive and finite, and both
villages must actually hold what they are giving away.

## The adaptive brain

Each village owns a `NegotiationBrain`, a small linear model trained online by
`scikit-learn`'s `SGDRegressor` through `partial_fit`. Every day it receives one
observation per commodity and updates its own weights, so the price a village is
willing to pay drifts with the scarcity it actually experiences.

From that prediction it derives:

```
reservation_price = base_value * max(floor, 1 + gain * scarcity - discount * surplus)
```

The scarcity it predicts is what lets the two sides disagree about a fair price,
which is what makes the bargaining worth simulating.

## The negotiation

Each side knows only its own reservation prices, so neither can simply accept
the other's number.

- The proposer opens at the reference rate both sides can compute from base
  values.
- The responder takes it, or answers with the lowest rate it can live with.
- Whatever is still contested is settled at the geometric middle of the two
  limits, which is the Nash bargaining point and never favours one side over
  the other.

The quantity is capped by the surplus on both sides, so no village ever gives
away what it needs to survive the coming days.

## Steering the world

The chat box takes a description, in plain language, and the model works out what
to change:

```
> make it harsh: the mines run dry and nobody meets anyone
Miners production.Minerals: 20 -> 0
trade_chance: 0.6 -> 0
> on day 3 there is a drought, the farmers grow almost nothing
held for day 3, the drought: production.Agriculture
```

Every change is shown as the value it replaced and the value it became, so what a
request actually did is visible rather than only where it left things. A request
that asks for what is already the case says so instead of claiming a change.

The tables follow along with every change. A `SETTINGS` table holds the values a
request can move, one row per village plus one for the rules of the run, and it is
redrawn the moment a change lands, including before the first day has been
simulated. Holdings sit in `STOCK`, and what the world made and ate on the day it
last ran sits in `LAST DAY`, which is empty until a day has happened rather than
showing a day that never ran.

A request can name a day, in which case the change is held in the world's calendar
and lands at the start of that day, before anybody produces.

**What may be changed, and how far it may be pushed:**

| Target  | Setting                             | Range           |
| ------- | ----------------------------------- | --------------- |
| world   | `time_per_day`                      | 0 to 24         |
| world   | `trade_chance`                      | 0 to 1          |
| world   | `max_rounds`                        | 1 to 10         |
| village | `population`                        | 1 to 1000       |
| village | `reserve_days`                      | 0 to 10         |
| village | `production_noise`                  | 0 to 1          |
| village | `production.<commodity>`            | 0 to 1000       |
| village | `consumption.<commodity>`           | 0 to 1000       |
| village | `stock.<commodity>`                 | 0 to 1000000    |

`consumption` is units per person per day. `production` is units per day for the
whole village, so a village of ten that makes twenty a day and uses two each
produces nothing net. Every commodity is an input to work rather than food to any
one village, and a shortage of any of them is now expressible rather than only a
harvest.

A value outside its range is pulled back to the nearest end and the correction is
reported, because a silent correction leaves the reader believing the world ended
up somewhere nobody asked for. A setting that does not exist, a village that is
not in the world and a value that is not a number are each dropped with a reason
that lists what is available, and the rest of the request is still carried out.

**A request is refused rather than approximated.** A tax rate is not a harvest, a
morale is not a population and hunger is not a shortage of stores. When part of a
request has no setting behind it, the narrator changes nothing for that part and
names it, so a world never moves somewhere nobody asked for:

```
> give the Miners a tax rate of ten percent
Not done:
I could not make sense of 'a tax rate for the Miners' as anything I can change.
```

**The seed is not changeable.** Changing a harvest changes the story; rewriting the
seed only destroys the ability to tell the same story twice.

When the model answers with prose instead of changes, the reply shows the
instruction back, shows what came back instead, and asks for it again.

## The narrator

A narrator watches the world and writes what happened, a few lines per day. It
never sees the simulation itself. It receives a `WorldSummary` holding only what a
story needs: the day, how the three villages are doing, the barters that took
place, and who came up short. The stock tables are left out on purpose, the
interface shows them separately and they would only crowd the context.

The last line is always a verdict on the day as a whole, one of:

| Situation                    | Closing verdict                                     |
| ---------------------------- | --------------------------------------------------- |
| every village fully supplied | `the world in balance`                              |
| one village short a little   | `a shortage of Handicrafts`                         |
| one village badly short      | `a severe shortage of Handicrafts`                  |
| two or three short of the same thing | `a shortage of Handicrafts shared by Farmers and Miners` |
| nobody short, stores far past what the world can spend | `a surplus of Minerals held by Farmers and Miners` |

There is deliberately no state in between: a village is either at its target
stock or it is missing something, so the world is either in balance or short. When
one village runs low that is its own bad luck and the verdict names it alone; when
several run low of the same commodity at once, that is a shortage the world cannot
trade its way out of, and the verdict names all of them together.

A pile-up is only reported when nobody is short, because goods in the stores are a
problem the day has already solved. It is measured across every village at once,
counting only units above a village's own target: the same goods spread thinly
between three stores are a pile the world cannot move, while the same goods in one
granary are just a village that made more than it needed. A pile has to reach four
days of world use before it is worth naming.

A run always has a language model behind it. There is no template writer, so a day
with no model gets no story rather than a badly written one.

The model is reached over `/v1/chat/completions`, so llama.cpp, Ollama, vLLM and
LM Studio all work against the same code.

With llama.cpp there is nothing to start by hand. The simulation launches the
server, waits for the model to finish loading, runs, and shuts the server down
again on the way out, including when you interrupt it:

```bash
python run.py --days 5
```

It picks `vendor/llama.cpp/build/bin/llama-server` and the largest model in
`models/`. A server already listening on the port is reused as it is instead of
being started twice.

| Variable                  | Meaning                                    | Default     |
| ------------------------- | ------------------------------------------ | ----------- |
| `WORLD_NARRATOR_BINARY`   | Path to the model server                    | vendored    |
| `WORLD_NARRATOR_MODEL_FILE` | Model to load                             | largest in `models/` |
| `WORLD_NARRATOR_HOST`     | Address the server binds to                 | `127.0.0.1` |
| `WORLD_NARRATOR_PORT`     | Port to listen on                           | `8080`      |
| `WORLD_NARRATOR_LOG`      | Where the server writes its own output      | discarded   |
| `WORLD_NARRATOR_BASE_URL` | Server to run yourself; setting it skips the bundled one | `127.0.0.1:8080` |
| `WORLD_NARRATOR_MODEL`    | Model name to ask for, when the server needs one | `local` |
| `WORLD_NARRATOR_API_KEY`  | Bearer token, when the server wants one     | none      |
| `WORLD_NARRATOR_GPU_LAYERS` | Layers to keep in video memory, `all` for every one | processor |
| `WORLD_NARRATOR_GPU_SPLIT` | How to spread the model over cards: `layer`, `row`, `tensor`, `none` | `layer` |
| `WORLD_NARRATOR_GPU_MAIN` | Which card to treat as the main one, counting from zero | all of them |

**Running on a card.** The bundled `llama-server` has to have been built with a
backend for the card, which is part of building llama.cpp rather than part of this
project. Without one, the run stays on the processor and says so rather than
quietly being slow. Asking for a card that the binary cannot use is refused up
front, because the server does not explain that itself.

```bash
cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release
WORLD_NARRATOR_GPU_LAYERS=all python run.py --days 30
```

Check what a binary can do before trusting it:

```bash
ls vendor/llama.cpp/build/bin/ | grep -E "cuda|hip|vulkan|metal"
```

A `libggml-cuda.so`, `libggml-hip.so`, `libggml-vulkan.so` or `libggml-metal.so`
sitting beside the binary means there is a card backend to use.

Pointing at a server you run yourself instead, which skips starting the bundled
one:

```bash
vendor/llama.cpp/build/bin/llama-server -m models/gemma-4-E2B-it-Q4_K_M.gguf --port 8080 &
WORLD_NARRATOR_BASE_URL=http://127.0.0.1:8080 python run.py
```

Rebuilding the server after a source update:

```bash
cd vendor/llama.cpp
CCACHE_DIR=/tmp/ccache cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j "$(nproc)"
```

`CCACHE_DIR` is required, otherwise the build fails with a read-only file system
error from the compiler cache rather than anything to do with the source.

`WORLD_NARRATOR_MODEL` is only needed when the server hosts more than one model,
and `WORLD_NARRATOR_API_KEY` is sent as a bearer token when the server asks for
one. Reasoning models such as Qwen3.5 spend their budget thinking and return an
empty answer, so thinking is switched off by default; set `thinking=True` on the
narrator if you want it back.

**The numbers and the commodity names are facts.** A model likes to turn `20.0`
into `twenty` and `Minerals` into `ores`, so every reply is checked against the
summary it was given. When a quantity or a commodity name went missing the reply
is refused and the day is reported as unwritten, rather than quietly replaced by
something nobody chose. A narrator that fails never stops the world: the day is
simulated either way, and the failure is shown in place.

Measured on four days of the default world, checking that every trade kept its
direction and its numbers:

| Model                  | Size  | Seconds per day | Direction kept | Numbers kept |
| ---------------------- | ----- | --------------- | -------------- | ------------ |
| `gemma-4-E2B-it`       | 3.4G  | 7               | all            | all          |
| `gemma-4-12B-it`       | 7G    | 36              | 11 of 12       | all          |
| `Qwen3.5-4B`           | 2.7G  | 16              | all            | 2 of 4 wrong |
| `Qwen3.5-2B`           | 1.5G  | 9               | reversed       | all          |

`gemma-4-E2B-it` is the pick: half the size of the 12B model, five times faster,
and the only one that never lost a direction. Use a model of at least 4B; a 2B
model mixes up who gave what. On CPU, expect roughly 7 to 20 seconds per day,
which suits the step at a time mode far better than a long batch.

Holding the numbers costs the model some freedom: asked to keep the digits, the
same model starts keeping the shape of the input sentences as well. The prose
is plainer but it is always right.

## Balance

Under ideal conditions the three villages converge on an even distribution:
every village ends fully supplied, and the spread in living standard between the
best fed and the worst fed falls to zero. A village holds one day of use
back before anything is tradable, which is what stops the barters from draining
a village below survival.

## Layout

| Module            | Responsibility                                  |
| ----------------- | ----------------------------------------------- |
| `world/goods.py`  | The three commodities and their reference prices. |
| `world/agent.py`  | `NegotiationBrain`, the online pricing model.     |
| `world/village.py`| Production, use and `trade`.                      |
| `world/negotiation.py` | Offers, counter offers and settlement.       |
| `world/world.py`  | `advance_day`, the tick that moves time forward.  |
| `world/logbook.py`| `DayEntry`, `RunLog` and the log writer.          |
| `world/tui.py`    | The full screen interface.                      |
| `world/narrator.py` | `WorldSummary`, the verdict and the narrator.  |
| `world/llm.py`      | The one place that talks to a model.           |
| `world/director.py` | Reads a change request and applies what is safe. |
| `world/server.py`  | Starts, waits for and stops the model server.  |
| `run.py`          | Command line entry point.                         |

## Development

```bash
python -m pytest    # tests
ruff check .        # lint
```
