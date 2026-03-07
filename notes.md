# Overcooked Model Notes



```
  XXPXX
  O1  O
  X  2X
  XDXSX
```


## TODO

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
- [ ] joint policies (`InCo`)

---

## Parameters

```python
A_NOISE_LEVEL = 0.001
B_NOISE_LEVEL = 0.0

INTERACT_SUCCESS_PROB = 1.0
```







