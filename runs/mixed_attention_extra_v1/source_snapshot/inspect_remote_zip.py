"""Read the official archive directory and small annotation metadata via HTTP ranges."""
import io,json,urllib.request
from zipfile import ZipFile
from pathlib import Path
from prepare_data import ROOT,save_json

class RemoteZip(io.RawIOBase):
    def __init__(self,url,length):self.url=url;self.length=length;self.position=0
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        self.position=offset if whence==0 else (self.position+offset if whence==1 else self.length+offset)
        return self.position
    def read(self,n=-1):
        if n<0:n=self.length-self.position
        n=min(n,self.length-self.position)
        if n<=0:return b''
        assert n<8*1024*1024,'Only small metadata requests permitted'
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={self.position}-{self.position+n-1}','User-Agent':'TomatoCNN-research'})
        with urllib.request.urlopen(request,timeout=30) as response:
            assert response.status==206
            data=response.read()
        assert len(data)==n
        self.position+=n
        return data

if __name__=='__main__':
    source=json.loads((ROOT/'data/sources/zenodo15868289.json').read_text())
    info=source['files'][0]
    with ZipFile(RemoteZip(info['links']['self'],info['size'])) as z:
        names=z.namelist();save_json(ROOT/'data/tomato_disease/archive_members_preview.json',names)
        print('Members',len(names),'examples',names[:20],flush=True)
        metadata=[p for p in names if Path(p).suffix.lower() in {'.yaml','.yml','.json','.names','.xml'}]
        print('Metadata examples',metadata[:20],flush=True)
        for name in metadata[:3]:
            text=z.read(name).decode();print(name,text[:4000],flush=True)
            (ROOT/'data/tomato_disease'/('preview_'+Path(name).name)).write_text(text,encoding='utf-8')
