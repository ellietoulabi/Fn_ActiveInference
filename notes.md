# Notes



```
  XXPXX
  O1  O
  X  2X
  XDXSX
```


## Status: 
 - semantic policies work




## TODO
- [ ] make sure everything is correct
- [ ] action-level policy
- [ ] dynamic policy

### model_init.py
- [x] states
- [x] observations
- [x] state–state dependencies
- [x] state–obs dependencies
- [x] add other position and other held object (`InDe`)

---

### A.py
- [x] soup delivered observation depends only on ck_delivered state
- [ ] other position and held observations (`InDe`)

---

### B.py
- [x] drop on full counter handled
- [ ] pickup from counter
- [x] model counters when agent interacts with them
- [ ] other position and held transitions (`InDe`)
- [ ] joint policy transitions (`InCo`)

---

### C.py
- [ ] (not implemented yet) (`InDe`)

---

### D.py
- [ ] other position and held initial states (`InDe`)

---

### Policies
- [x] joint policies (`InCo`)
- [x] Fixed Predefined set of policies saved in a file
- [ ] think about if you need to change Active Inference implementation
- [ ] list all policy ideas (why james' doesnt work)
- [ ] change the gen model
- [ ] run
- [ ] REMEMBER to save results to discuss them in
---

## Parameters

```python
A_NOISE_LEVEL = 0.001
B_NOISE_LEVEL = 0.0

INTERACT_SUCCESS_PROB = 1.0
```







