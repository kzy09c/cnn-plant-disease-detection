"""Inspect only the official PlantWild archive directory via byte ranges."""
import json
from pathlib import Path
from collections import Counter
from zipfile import ZipFile
from inspect_remote_zip import RemoteZip
from prepare_data import ROOT,DATA,save_json

if __name__=='__main__':
    tree=json.loads((DATA/'sources/uqtwei2_PlantWild_tree.json').read_text())
    file=next(x for x in tree if x['path']=='plantwild.zip')
    url='https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip'
    with ZipFile(RemoteZip(url,file['size'])) as z:
        names=z.namelist()
        selected=[i for i in z.infolist() if 'tomato' in i.filename.lower() and not i.is_dir()]
        print('Tomato entries',len(selected),'directories',Counter(str(Path(x.filename).parent) for x in selected),flush=True)
        print('Metadata files',[n for n in names if Path(n).suffix.lower() in ['.txt','.json','.csv']],flush=True)
        out=DATA/'plantwild';out.mkdir(exist_ok=True)
        save_json(out/'archive_index.json',[{'filename':i.filename,'header_offset':i.header_offset,'compress_size':i.compress_size,'file_size':i.file_size,'CRC':i.CRC,'compress_type':i.compress_type,'extra_length':len(i.extra)} for i in selected])
        save_json(out/'archive_source.json',{'url':url,'hf_file_metadata':file,'license':'CC BY-NC-ND 4.0; local noncommercial research; original images unmodified and not redistributed'})
        for name in [n for n in names if Path(n).suffix.lower() in ['.txt','.json','.csv']][:10]:
            info=z.getinfo(name)
            if info.file_size>8*1024*1024:continue
            b=z.read(name);(out/Path(name).name).write_bytes(b)
            print(name,'bytes',len(b),'preview',b[:600].decode(errors='replace'),flush=True)
