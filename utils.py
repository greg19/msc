import csv
from typing import Any
import pandas as pd


def binary_encoding(ids: list[int], d: int) -> list[bool]:
    return [i in ids for i in range(d)]


def parse_file(path: str):
    with open(path, 'r', newline='', encoding="utf-8") as csvfile:
        meta = {}
        projects = {}
        votes = {}
        section = ""
        header = []
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            if str(row[0]).strip().lower() in ["meta", "projects", "votes"]:
                section = str(row[0]).strip().lower()
                header = next(reader)
            elif section == "meta":
                meta[row[0]] = row[1].strip()
            elif section == "projects":
                projects[row[0]] = {}
                for it, key in enumerate(header[1:]):
                    projects[row[0]][key.strip()] = row[it+1].strip()
            elif section == "votes":
                votes[row[0]] = {}
                for it, key in enumerate(header[1:]):
                    votes[row[0]][key.strip()] = row[it+1].strip()
    return meta, projects, votes

def load_pb(path: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    meta, projects, votes = parse_file(path)
    
    project_id_map = {x: i for i, x in enumerate(projects.keys())}
    
    for k in votes.keys():
        votes[k]['vote'] = sorted([project_id_map[p] for p in votes[k]['vote'].split(',')])
    
    meta['num_projects'] = int(meta['num_projects'])
    meta['num_votes'] = int(meta['num_votes'])
    meta['budget'] = int(meta['num_votes'])
    meta['min_length'] = int(meta['min_length'])
    meta['max_length'] = int(meta['max_length'])

    projects = {project_id_map[k]: v for k, v in projects.items()}

    projects_df = pd.DataFrame.from_dict(projects, orient='index')
    projects_df = projects_df.convert_dtypes()
    projects_df = projects_df.astype({
        'cost': 'int32',
        'votes': 'int32',
        'selected': 'bool',
    })
    projects_df['category'] = [set(x.split(',')) for x in projects_df['category']]
    projects_df['target'] = [set(x.split(',')) for x in projects_df['target']]
    projects_df['longitude'] = pd.to_numeric(projects_df['longitude'])
    projects_df['latitude'] = pd.to_numeric(projects_df['latitude'])

    votes_df = pd.DataFrame(votes.values())
    votes_df = votes_df.convert_dtypes()

    return meta, projects_df, votes_df