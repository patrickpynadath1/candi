# Dedicated experiment queues on one Ray cluster

Ray node labels select which worker may execute a task. GPU resource requests
still reserve a GPU on that worker. A queue label alone does not reserve a GPU
or bind a physical device; start each worker with its assigned
`CUDA_VISIBLE_DEVICES` and `--num-gpus=1`.

For nested training/inference, add to each manifest case:

```yaml
resources:
  cpu: 4
  gpu: 1
  label_selector:
    queue: geometry
```

Both training and inference workers use that selector. Omit it to allow normal
Ray resource-based scheduling. Existing queue contents remain the experiment
manifest plus persisted Optuna/results state, not a separate queue database.

For a standalone resolved training config:

```bash
uv run --locked --extra tuning --extra scoring --extra tracking \
  python -m core.training.submit training.yaml \
  --address 10.145.21.37:34930 --working-dir /path/to/frozen/source \
  --queue latent-text8
```

Use the existing Ray token environment before importing Ray. Never embed tokens
in manifests. Keep the frozen source and resolved config alongside launch records.

The current server has a CPU-only head and three worker processes on the same
physical host. Start worker nodes sequentially: Ray chooses local socket names
by checking the shared session directory, so simultaneous startup can race.
Give each worker a unique node name, dashboard-agent port, and nonoverlapping
worker-port range. `CUDA_VISIBLE_DEVICES` is set before starting its raylet.
Do not use global `ray stop` on this shared server; stop only the recorded process
trees for the intended cluster.

Current allocation:

| Queue label | Physical GPU | Work |
| --- | --- | --- |
| geometry | 0 | Three unfinished 16-layer geometry runs |
| scaling | 5 | Sixteen E2D2 depth cases and two raw-context controls |
| depth-mixture | 3 | Six encoder-K/V mixture cases |


Cluster address: `10.145.21.37:34930`. Exact startup commands, PIDs, submission
commands, original queue inventory and a label-to-device probe are under
`/home/patrick/.cache/duo/plans/ray-gpu-reallocation-20261101/`:
`shared-cluster.json`, `shared-submissions.json`, `inventory.json`, `label-probe.json`.
Existing completed results are retained in original campaign directories.
Restarted sweep outputs end in `-restart-gpu013-labeled`; the text8 pilot output
ends in `-20261101-labeled`. The old isolated clusters and direct pilot are stopped.

The latent-autoencoder jobs were stopped at user request on November 2.
GPU 1 was released; the scaling queue moved to GPU 5. Existing checkpoints and
results were retained. Historical allocation probes describe the old placement;
use shared-cluster.json and shared-submissions.json for the current allocation.
