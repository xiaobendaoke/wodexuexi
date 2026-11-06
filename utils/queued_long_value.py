"""Expanded public action bank and fixed-unit deadline/contact features."""
from utils.queued_service_shared_budget import BudgetAdapter
from utils.queued_joint_opportunity import expanded_bank,conditional_suffix
from environment.queued_service import distance

FEATURES=119


class LongAdapter(BudgetAdapter):
    def __init__(self,expanded=True):super().__init__();self.expanded=expanded;self.old_count=None

    def prepare(self,obs):
        super().prepare(obs);self.old_count=len(self.plans)
        if self.expanded:self.plans=expanded_bank(obs,self.plans)

    def motion(self,obs):
        bank=super().motion(obs);p=obs['protocol'];now=obs['time'];bins=[0.]*18
        for task in obs['tasks']:
            if task['target'] is None:continue
            slack=task['deadline']-now;b=0 if slack<=1 else 1 if slack<=2 else 2
            bins[task['target']*3+b]+=task['remaining_cycles']/p.capacity[task['target']]
        rows=[]
        for i,action in enumerate(bank['actions']):
            contact=[0.]*15
            for task in obs['tasks']:
                source=task['source']
                if source is None or task['remaining_upload']<=0:continue
                ue=obs['ues'][task['ue']];a=obs['positions'][source];d=action[source]
                fraction=sum(distance(ue,(a[0]+d[0]*(j+.5)/10,a[1]+d[1]*(j+.5)/10))<=p.coverage for j in range(10))/10
                slack=task['deadline']-now;b=0 if slack<=1 else 1 if slack<=2 else 2
                contact[source*3+b]+=task['remaining_upload']/20e6*fraction
            rows.append(bank['features'][i]+bins+contact)
        bank['features']=rows+[[0.]*FEATURES for _ in range(16-len(rows))]
        assert all(len(row)==FEATURES for row in bank['features'])
        return bank


def remaining_future(workload,ues,start,replica):
    """Original conditional arrival law until t=200; future is simulator-only."""
    result={}
    for beginning in range(start,200,5):
        for offset,row in enumerate(conditional_suffix(workload,ues,beginning,replica),1):result[beginning+offset]=row
    return result
