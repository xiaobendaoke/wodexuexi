"""Read active BibTeX citations and cache authoritative Crossref metadata.

The query report does not certify claim support or automatically replace a
bibliography. Publisher/full-text comparison remains a separate human check.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import html
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paths import REPO_ROOT


def entries(text):
    out = {}
    for match in re.finditer(r'@(\w+)\s*\{\s*([^,]+),',text):
        start = match.end()
        depth, pos = 1, start
        while depth and pos < len(text):
            if text[pos] == '{': depth += 1
            elif text[pos] == '}': depth -= 1
            pos += 1
        fields = {}
        body = text[start:pos-1]
        for fm in re.finditer(r'(\w[\w-]*)\s*=\s*\{',body):
            i, j, nesting = fm.end(), fm.end(), 1
            while nesting and j < len(body):
                if body[j] == '{': nesting += 1
                elif body[j] == '}': nesting -= 1
                j += 1
            fields[fm.group(1).lower()] = body[i:j-1]
        out[match.group(2).strip()] = dict(type=match.group(1),fields=fields,
            start=match.start(),end=pos)
    return out


def normalized(text):
    return re.sub(r'[^\w\u4e00-\u9fff]','',html.unescape(re.sub('<[^>]*>','',text)).lower())


def query(key, entry):
    fields = entry['fields']
    title = fields.get('title','').replace('{','').replace('}','')
    doi = fields.get('doi')
    if doi:
        url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi,safe='')
    else:
        params = urllib.parse.urlencode({'query.title':title,'rows':3})
        url = 'https://api.crossref.org/works?' + params
    try:
        request = urllib.request.Request(url,headers={'User-Agent':'ThesisReferenceVerification/1.0 (academic metadata inspection)'})
        with urllib.request.urlopen(request,timeout=30) as response:
            msg=json.load(response)['message']
        items=[msg] if doi else msg['items']
        candidates=[]
        for item in items:
            got=item.get('title',[''])[0]
            similarity=SequenceMatcher(None,normalized(title),normalized(got)).ratio()
            candidates.append(dict(similarity=similarity,title=got,DOI=item.get('DOI'),
                author=item.get('author',[]),year=item.get('published',{}).get('date-parts'),
                journal=item.get('container-title'),volume=item.get('volume'),issue=item.get('issue'),
                page=item.get('page'),article_number=item.get('article-number'),type=item.get('type'),
                URL=item.get('URL'),link=item.get('link',[])))
        candidates.sort(key=lambda d:d['similarity'],reverse=True)
        return key,dict(original=fields,query_url=url,candidates=candidates,
            metadata_match=bool(candidates and candidates[0]['similarity']>.92),
            claim_support_not_automatically_verified=True)
    except Exception as exc:
        return key,dict(original=fields,query_url=url,error=str(exc),metadata_match=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bib',type=Path,default=REPO_ROOT/'latex/reference.bib')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    opts=p.parse_args()
    cited=set()
    for path in (REPO_ROOT/'latex/docs').glob('*.tex'):
        for group in re.findall(r'\\cite\w*\*?(?:\[[^]]*\])*\{([^}]*)\}',path.read_text()):
            cited.update(k.strip() for k in group.split(','))
    library=entries(opts.bib.read_text())
    records=json.loads(opts.output.read_text()) if opts.output.exists() else {}
    pending=[k for k in sorted(cited) if k in library and (k not in records or records[k].get('original')!=library[k]['fields'])]
    opts.output.parent.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        tasks=[pool.submit(query,key,library[key]) for key in pending]
        for task in as_completed(tasks):
            key,result=task.result();records[key]=result
            opts.output.write_text(json.dumps(records,ensure_ascii=False,indent=2))
            print(key, 'metadata match' if result['metadata_match'] else 'review required',flush=True)
    print('Active citations:',len(cited),'Missing keys:',sorted(cited-library.keys()))


if __name__=='__main__':main()
