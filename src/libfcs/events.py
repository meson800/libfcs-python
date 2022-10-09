"""
Main FlowEvents object.
"""

from typing import Dict, Iterable, List, Literal, Union
import copy
import warnings

import pandas as pd

from ._libfcs_ext import FCS # type: ignore
from .gates import Gate

class FlowEvents:
    events: pd.DataFrame
    metadata: pd.DataFrame
    _underlying_fcs: List[FCS]

    def __init__(self, fcs_files: Iterable[FCS],
                *,
                param_name_preference: Union[Literal['full'], Literal['short']] = 'full'
                ):
        # Setup FCS metadata dataframe
        self.metadata = pd.DataFrame([], columns=[
            'acquire_start', 'acquire_end',
            'cell_description', 'comment',
            'cytometer_type', 'cytometer_serial_number',
            'institution', 'experimenter', 'operator', 'original_filename',
            'last_modified', 'last_modifier', 'plate_id', 'plate_name',
            'project', 'specimen', 'specimen_source', 'computer_name', 'well_id'
        ])
        self._underlying_fcs = []
        dfs: List[pd.Dataframe] = []
        for idx, fcs in enumerate(fcs_files):
            self._underlying_fcs.append(fcs)
            # Set metadata for this file
            self.metadata[idx] = {
                'acquire_start': pd.Timestamp(f'{fcs.acquire_date_str} {fcs.acquire_time_str}'),
                'acquire_end': (
                    pd.Timestamp(f'{fcs.acquire_date_str} {fcs.acquire_end_time_str}')
                    + (pd.Timedelta(1, 'day')
                       if pd.Timestamp(fcs.acquire_end_time_str) < pd.Timestamp(fcs.acquire_time_str)
                       else pd.Timedelta(0))),
                'cell_description': fcs.cells,
                'comment': fcs.comment,
                'cytometer_type': fcs.cytometer_type,
                'cytometer_serial_number': fcs.cytometer_serial_number,
                'institution': fcs.institution,
                'experimenter': fcs.experimenter,
                'operator': fcs.operator,
                'original_filename': fcs.original_filename,
                'last_modified': pd.Timestamp(fcs.last_modified_str),
                'last_modifier': fcs.last_modifier,
                'plate_id': fcs.plate_id,
                'plate_name': fcs.plate_name,
                'project': fcs.project,
                'specimen': fcs.specimen,
                'specimen_source': fcs.specimen_source,
                'computer_name': fcs.computer,
                'well_id': fcs.well_id
            }
            # Get events
            events = pd.DataFrame(fcs.uncompensated, columns=[
                p.name if param_name_preference == 'full' and p.name is not None
                else p.short_name
                for p in fcs.parameters
            ])
            events['m_idx'] = idx
            dfs.append(events)
        self.events = pd.concat(dfs, ignore_index=True)

    def __repr__(self):
        return f"{len(self.events)} FCS events"
    
    def _repr_html_(self):
        return ""
    
    def filter(self, gate: Gate, *, level=True) -> 'FlowEvents':
        """
        Filters the included events, returning a new FCSEvents object
        """
        filtered = copy.copy(self)
        filtered.events = self.events[gate.inclusion(self.events, level)].copy()
        return filtered

    def label(self, label_name: str, gate: Gate) -> 'FlowEvents':
        """
        Adds an additional column to the events dataframe, using the given gate.
        """
        labeled = copy.copy(self)
        labeled.events = self.events.copy()
        labeled.events[label_name] = gate.label(labeled.events)
        return labeled
    
    def describe(self, gate: Gate) -> pd.DataFrame:
        return self.events
    
