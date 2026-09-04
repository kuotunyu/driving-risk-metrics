# Published evidence

`bdd100k_semseg_v1/` holds the immutable formal run index and every analysis
document the release cites: the metric table, the paired intervals, the ranking
comparison, the failure-gallery manifest and the extended metrics. Each is
produced by a `driving-risk` command and none is ever edited by hand.

They are tracked here, under `docs/`, rather than under `artifacts/`, because
`/artifacts/` is ignored: `driving-risk audit-claims` resolves every claim's
`artifact_path` relative to the repository root, and in a clean clone an ignored
path does not exist. Evidence a release rests on has to survive a clone.

The nine runs themselves — checkpoints and roughly 54 GiB of per-image
prediction artifacts — stay outside Git. They are the input to these documents,
not part of the release.
