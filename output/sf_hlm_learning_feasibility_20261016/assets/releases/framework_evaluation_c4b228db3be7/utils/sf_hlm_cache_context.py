"""Summaries of local cache options already visible in the shared public pool."""
from utils.sf_hlm_cache_calibration import own_cache


def public_context(rows):
    groups={str(j):dict(requests=0,visible=0,hits=0) for j in range(5)}
    total=dict(requests=0,visible=0,hits=0)
    for row in rows:
        source=row['source']
        if isinstance(source,bool) or not isinstance(source,int) or source not in range(5):raise ValueError('Invalid public source')
        value,visible=own_cache(row)
        if visible and value not in (0.,1.):raise ValueError('Visible local cache presence must be binary')
        for group in (total,groups[str(source)]):
            group['requests']+=1;group['visible']+=int(visible);group['hits']+=int(value) if visible else 0
    for group in (total,*groups.values()):
        group['fraction']=group['hits']/group['visible'] if group['visible'] else None
        group['all_visible_miss']=bool(group['visible'] and not group['hits'])
    return dict(global_context_contract='observed_local_options_only_v1',**{'global':total},sources=groups)
