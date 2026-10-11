from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .lanl import iter_redteam


@dataclass(frozen=True)
class IncidentEpisode:
    episode_id: int
    start_time: int
    end_time: int
    hidden_origin: str
    first_user: str
    first_lateral_source: str
    destinations: tuple[str, ...]
    event_count: int


def build_episodes(redteam_path: str | Path, gap_seconds: int = 900, minimum_events: int = 2) -> list[IncidentEpisode]:
    events = sorted(iter_redteam(redteam_path), key=lambda item: item[0])
    grouped: list[list[tuple[int, str, str, str]]] = []
    for event in events:
        if not grouped or event[0] - grouped[-1][-1][0] > gap_seconds:
            grouped.append([event])
        else:
            grouped[-1].append(event)
    episodes: list[IncidentEpisode] = []
    for group in grouped:
        destinations = tuple(dict.fromkeys(event[3] for event in group))
        if len(group) < minimum_events or len(destinations) < 2:
            continue
        first = group[0]
        episodes.append(
            IncidentEpisode(
                episode_id=len(episodes),
                start_time=first[0],
                end_time=group[-1][0],
                hidden_origin=first[3],
                first_user=first[1],
                first_lateral_source=first[2],
                destinations=destinations,
                event_count=len(group),
            )
        )
    return episodes


def chronological_split(episodes: list[IncidentEpisode], fractions=(0.6, 0.2, 0.2)):
    first = int(len(episodes) * fractions[0])
    second = int(len(episodes) * (fractions[0] + fractions[1]))
    return {"train": episodes[:first], "validation": episodes[first:second], "test": episodes[second:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redteam", required=True)
    parser.add_argument("--gaps", nargs="+", type=int, default=[900, 1800, 3600, 7200])
    args = parser.parse_args()
    report = {}
    for gap in args.gaps:
        incidents = build_episodes(args.redteam, gap)
        report[str(gap)] = {
            "episodes": len(incidents),
            "unique_origins": len({item.hidden_origin for item in incidents}),
            "events": sum(item.event_count for item in incidents),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
