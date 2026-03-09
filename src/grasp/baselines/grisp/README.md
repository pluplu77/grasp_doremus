# GRISP: Guided Recurrent IRI Selection over SPARQL Skeletons

GRISP is a question-answering method for arbitrary knowledge graphs
that works by fine-tuning and running a small large language model
in a generate-then-retrieve fashion. The retrieval is performed
iteratively using search and list-wise re-ranking, and guided by
the model's own generated SPARQL skeletons and the knowledge graph structure.

You can also [download](https://ad-publications.cs.uni-freiburg.de/grisp/)
and run our fine-tuned models for Freebase and Wikidata.

## Installation

See the [main README](../../../../README.md) for installation instructions
of GRASP. Then make sure you have the GRISP extension installed as well:

```bash
pip install -e ".[grisp]"
```

> Note: All commands shown below have useful additional options that
> are not shown here for simplicity. Please check the help messages
> of the respective commands for more details.

## Train a GRISP model

An exemplary training config with the most important training options
is provided [`here`](../../../../configs/grisp/train.yaml).

We will show a running example with the WikiWebQuestions dataset for
Wikidata. Make sure that you have set up the corresponding GRASP
indices for the knowledge graph you want to work with before
starting with the steps below. Again, see the [main README](../../../README.md)
for instructions on how to do this.

1: Prepare your data in jsonl format

```bash
# see an example of the data format
cat data/benchmark/wikidata/wwq/train.jsonl | head -1 | jq
```

2: Convert the data to the GRISP format

```bash
# convert training data
python -m grasp.baselines.grisp.data \
  wikidata \
  data/benchmark/wikidata/wwq/train.jsonl \
  data/grisp/wikidata/wwq/train.jsonl

# same for validation data
python -m grasp.baselines.grisp.data \
  wikidata \
  data/benchmark/wikidata/wwq/val.jsonl \
  data/grisp/wikidata/wwq/val.jsonl
```

3: Materialize the GRISP training data (Optional but recommended)

GRISP performs online data preprocessing and augmentation during training,
which is computationally expensive. However, you can also do this offline
and generate samples for a given number of epochs in advance. This is
recommended because it requires you to do it only once, and speeds up
training significantly.

```bash
# materialize training data for 4 epochs
python -m grasp.baselines.grisp.materialize \
  data/grisp/wikidata/wwq/train.jsonl \
  data/grisp/wikidata/wwq/train.materialized.jsonl \
  4 # number of epochs to materialize data for

# materialize validation data, which should stay the same across
# epochs, so we only need to do it for 1 epoch
# note: the --is-val flag is important to avoid data
# augmentations for validation data
python -m grasp.baselines.grisp.materialize \
  data/grisp/wikidata/wwq/val.jsonl \
  data/grisp/wikidata/wwq/val.materialized.jsonl \
  1 \
  --is-val
```

4: Train the model

Adapt the training config to your setup, then run the training command.
Most
If you have materialized the data, you should set the `materialized`
option to `true` and point the train and validation data paths
to the materialized data.

```bash
python -m grasp.baselines.grisp.train \
  configs/grisp/train.yaml \
  data/grisp/runs/my-wikidata-wwq-model # output directory
```

> Note: We use Wandb for experiment tracking by default, so
> set WANDB_PROJECT and WANDB_ENTITY environment variables
> to log your runs to your Wandb account.

## Run a GRISP model

An exemplary run config with the most important inference options
is provided [`here`](../../../../configs/grisp/run.yaml).

```bash
python -m grasp.baselines.grisp.run \
  configs/grisp/run.yaml \
  data/grisp/runs/my-wikidata-wwq-model \ # path to the trained model
  file \ # when evaluating on a dataset, use the file subcommand
  --input-file data/benchmark/wikidata/wwq/test.jsonl \ # path to test data
  --output-file data/grisp/runs/my-wikidata-wwq-model/predictions.jsonl
```

## Serve a GRISP model

```bash
python -m grasp.baselines.grisp.server configs/grisp/serve.yaml
```

The server runs on port 6790 by default and exposes the following HTTP endpoints.

| Endpoint | Method | Description |
|---|---|---|
| `/knowledge_graphs` | GET | Returns list with the configured KG name |
| `/config` | GET | Returns the server configuration |
| `/run` | POST | Run GRISP on a single question |

#### `POST /run`

**Request body:**

```json
{"question": "Where was Angela Merkel born?"}
```

**Response:** GRISP output as a JSON object

**Error codes:** `503` server busy, `504` generation timeout

## Evaluate a GRISP model

The `grisp.run` command produces outputs that are compatible
with the ones of GRASP. So you can just use the `evaluate`
subcommand of GRASP to evaluate.

```bash
# to evaluate the test predictions with f1 score
grasp evaluate f1 \
  data/benchmark/wikidata/wwq/test.jsonl \
  data/grisp/runs/my-wikidata-wwq-model/predictions.jsonl
```
