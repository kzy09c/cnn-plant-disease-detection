"""Read public Mendeley metadata for potential field datasets."""
import json
import urllib.request
from pathlib import Path
from inspect_sources import fetch,ROOT

if __name__=='__main__':
    for identifier in ['2bdfjb99k5','bpfd9cns5g']:
        for endpoint in [f'https://data.mendeley.com/public-api/datasets/{identifier}',f'https://data.mendeley.com/public-api/datasets/{identifier}/versions/1']:
            try:
                source=fetch(endpoint)
                (ROOT/'data/sources'/f'{identifier}.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
                print(identifier,'version',source.get('version'),'name',source.get('name'),flush=True)
                print('Files',len(source.get('files',[])),'bytes',sum(f.get('content_details',{}).get('size',0) for f in source.get('files',[])),flush=True)
                break
            except Exception as e:print(endpoint,repr(e),flush=True)
