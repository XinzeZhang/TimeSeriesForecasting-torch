import json
from data_process.util import Params

import argparse

dict_path = 'data/paper.esm/lag_settings.json'

x = {
  "datasets": {
      'AR1':{
          'lag_order': 15
      },
      'CrudeOilPrice.wti':{
          'lag_order':45
      }
  }
}
with open (dict_path,'w')  as f :
    json.dump(x,f,indent=4)
params = Params('',zero_start=True)
params.update(dict_path)


dataset = 'CrudeOilPrice.wti'
steps = params.datasets[dataset]['lag_order']
print(steps)

print()