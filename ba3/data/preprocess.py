from __future__ import annotations

import argparse
import gzip
import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

from .episodes import IncidentEpisode, build_episodes, chronological_split
from .lanl import resolve_modality_path


@dataclass
class EpisodeTelemetry:
    episode: IncidentEpisode
    pre_activity: dict[str, Counter] = field(default_factory=dict)
    post_activity: dict[str, Counter] = field(default_factory=dict)
    pre_in_degree: Counter = field(default_factory=Counter)
    pre_out_degree: Counter = field(default_factory=Counter)
    pre_alert_contacts: Counter = field(default_factory=Counter)
    post_alert_contacts: dict[str, Counter] = field(default_factory=dict)


_SCHEMA = {
    "auth": (["time", "source_user", "user", "source", "destination", "auth_type", "logon_type", "orientation", "status"], ["time", "source", "destination"]),
    "dns": (["time", "source", "destination"], ["time", "source", "destination"]),
    "flow": (["time", "duration", "source", "source_port", "destination", "destination_port", "protocol", "packets", "bytes"], ["time", "source", "destination"]),
    "proc": (["time", "user", "source", "destination", "action"], ["time", "source", "destination"]),
}


def aggregate_modality(records, path, modality, history_seconds, post_seconds):
    episodes = [record.episode for record in records]
    intervals = [(e.start_time - history_seconds, e.start_time + post_seconds, i) for i, e in enumerate(episodes)]
    max_time = max(end for _, end, _ in intervals)
    names, include = _SCHEMA[modality]
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=names, block_size=32 * 1024 * 1024, use_threads=True),
        convert_options=pacsv.ConvertOptions(
            include_columns=include,
            column_types={"time": pa.int64(), "source": pa.string(), "destination": pa.string()},
        ),
    )
    scanned = 0
    retained = 0
    for batch in reader:
        times = batch.column("time").to_numpy(zero_copy_only=False)
        if len(times) == 0:
            continue
        if int(times[0]) > max_time:
            break
        scanned += len(times)
        source = batch.column("source").to_pandas()
        destination = batch.column("destination").to_pandas()
        start, end = int(times[0]), int(times[-1])
        for lower_i, upper_i, episode_index in intervals:
            if lower_i > end or upper_i < start:
                continue
            record = records[episode_index]
            episode = record.episode
            relevant = (times >= lower_i) & (times <= upper_i)
            if not np.any(relevant):
                continue
            pre = relevant & (times < episode.start_time)
            post = relevant & ~pre
            alerts = set(episode.destinations[1:])
            if np.any(pre):
                pre_source = source[pre]
                pre_destination = destination[pre]
                record.pre_activity[modality].update(pre_source.value_counts().to_dict())
                record.pre_activity[modality].update(pre_destination.value_counts().to_dict())
                if modality == "auth":
                    record.pre_out_degree.update(pre_source.value_counts().to_dict())
                    record.pre_in_degree.update(pre_destination.value_counts().to_dict())
                    record.pre_alert_contacts.update(pre_source[pre_destination.isin(alerts)].value_counts().to_dict())
                    record.pre_alert_contacts.update(pre_destination[pre_source.isin(alerts)].value_counts().to_dict())
            if np.any(post):
                post_source = source[post]
                post_destination = destination[post]
                record.post_activity[modality].update(post_source.value_counts().to_dict())
                record.post_activity[modality].update(post_destination.value_counts().to_dict())
                record.post_alert_contacts[modality].update(post_source[post_destination.isin(alerts)].value_counts().to_dict())
                record.post_alert_contacts[modality].update(post_destination[post_source.isin(alerts)].value_counts().to_dict())
            retained += int(np.count_nonzero(relevant))
    return {"scanned": scanned, "retained_episode_assignments": retained}


def preprocess(data_dir, output, gap_seconds, history_seconds, post_seconds, modalities):
    episodes = build_episodes(resolve_modality_path(data_dir, "redteam"), gap_seconds=gap_seconds)
    records = [
        EpisodeTelemetry(
            episode=episode,
            pre_activity={name: Counter() for name in modalities},
            post_activity={name: Counter() for name in modalities},
            post_alert_contacts={name: Counter() for name in modalities},
        )
        for episode in episodes
    ]
    audit = {
        modality: aggregate_modality(records, resolve_modality_path(data_dir, modality), modality, history_seconds, post_seconds)
        for modality in modalities
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    splits = chronological_split(episodes)
    manifest = {
        "gap_seconds": gap_seconds,
        "history_seconds": history_seconds,
        "post_seconds": post_seconds,
        "modalities": modalities,
        "episodes": len(episodes),
        "unique_hidden_origins": len({item.hidden_origin for item in episodes}),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "audit": audit,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-seconds", type=int, default=900)
    parser.add_argument("--history-seconds", type=int, default=604800)
    parser.add_argument("--post-seconds", type=int, default=21600)
    parser.add_argument("--modalities", nargs="+", default=["auth", "dns", "flow", "proc"])
    args = parser.parse_args()
    print(json.dumps(preprocess(args.data_dir, args.output, args.gap_seconds, args.history_seconds, args.post_seconds, args.modalities), indent=2))


if __name__ == "__main__":
    main()
