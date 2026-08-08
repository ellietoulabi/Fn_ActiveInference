[Env] Using multi-agent env: layout=cramped_room

========================================================================
  IndividuallyCollective: two agents, cramped_room (seed=76)
========================================================================

  --- Step 1 ---
    Env state:  A0@0 held=none | A1@5 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾 [93m▲ [0m    🄾 
      🅲     [38;5;208m▲ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=none  facing=NORTH
    A1: pos(walkable)=5  holding=none  facing=NORTH
    Obs A0: self_pos=0 self_ori=0 self_held=0 other_pos=5 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=5 self_ori=0 self_held=0 other_pos=0 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.338:
        #1 [   W→I→N]                      0.010
        #2 [   W→I→S]                      0.010
        #3 [   W→I→E]                      0.010
        #4 [   W→I→W]                      0.010
        #5 [   W→I→S]                      0.010
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→W→I]                      0.005
        #3 [   W→W→S]                      0.005
        #4 [   W→W→W]                      0.005
        #5 [   I→I→I]                      0.005
    Action A0: SOUTH [1]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 2 ---
    Env state:  A0@3 held=none | A1@5 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾       🄾 
      🅲 [93m▼ [0m  [38;5;208m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=none  facing=SOUTH
    A1: pos(walkable)=5  holding=none  facing=SOUTH
    Obs A0: self_pos=3 self_ori=1 self_held=0 other_pos=5 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=5 self_ori=1 self_held=0 other_pos=3 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.199:
        #1 [   I→S→W]                      0.012
        #2 [   I→W→N]                      0.012
        #3 [   I→N→N]                      0.012
        #4 [   I→N→S]                      0.012
        #5 [   I→N→E]                      0.012
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→W→I]                      0.005
        #3 [   W→W→S]                      0.005
        #4 [   W→W→W]                      0.005
        #5 [   I→I→I]                      0.005
    Action A0: INTERACT [5]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 3 ---
    Env state:  A0@3 held=dish | A1@5 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾       🄾 
      🅲 [93m▼ [0m  [38;5;208m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=5  holding=none  facing=SOUTH
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=5 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=5 self_ori=1 self_held=0 other_pos=3 other_held=2 pot=0 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.334:
        #1 [   N→I→S]                      0.010
        #2 [   N→I→E]                      0.010
        #3 [   N→I→N]                      0.010
        #4 [   W→I→E]                      0.010
        #5 [   W→I→W]                      0.010
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→W→I]                      0.005
        #3 [   W→W→S]                      0.005
        #4 [   W→W→W]                      0.005
        #5 [   I→I→I]                      0.005
    Action A0: WEST [3]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 4 ---
    Env state:  A0@3 held=dish | A1@4 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾       🄾 
      🅲 [93m◀ [0m[38;5;208m◀ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=4  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=4 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=0 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.191:
        #1 [   I→S→S]                      0.011
        #2 [   I→S→W]                      0.011
        #3 [   I→W→S]                      0.011
        #4 [   I→W→W]                      0.011
        #5 [   I→N→N]                      0.011
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   W→S→N]                      0.005
        #4 [   W→I→N]                      0.005
        #5 [   W→S→I]                      0.005
    Action A0: INTERACT [5]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 5 ---
    Env state:  A0@3 held=none | A1@4 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾       🄾 
      🅿 [93m◀ [0m[38;5;208m▼ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=none  facing=WEST
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=3 self_ori=3 self_held=0 other_pos=4 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=3 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=0.01)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.354:
        #1 [   S→I→S]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→N]                      0.011
        #5 [   S→I→W]                      0.011
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   W→S→N]                      0.005
        #4 [   W→I→N]                      0.005
        #5 [   W→S→I]                      0.005
    Action A0: STAY [4]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 6 ---
    Env state:  A0@3 held=none | A1@4 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾       🄾 
      🅿 [93m◀ [0m[38;5;208m▼ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=none  facing=WEST
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=3 self_ori=3 self_held=0 other_pos=4 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=3 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   W→S→N]                      0.005
        #4 [   W→I→N]                      0.005
        #5 [   W→S→I]                      0.005
    Action A0: STAY [4]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 7 ---
    Env state:  A0@3 held=none | A1@5 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾       🄾 
      🅿 [93m◀ [0m  [38;5;208m▶ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=none  facing=WEST
    A1: pos(walkable)=5  holding=none  facing=EAST
    Obs A0: self_pos=3 self_ori=3 self_held=0 other_pos=5 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=5 self_ori=2 self_held=0 other_pos=3 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→W→I]                      0.005
        #3 [   W→W→S]                      0.005
        #4 [   W→W→W]                      0.005
        #5 [   I→I→I]                      0.005
    Action A0: EAST [2]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 8 ---
    Env state:  A0@4 held=none | A1@5 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾       🄾 
      🅿   [93m▶ [0m[38;5;208m▶ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=none  facing=EAST
    A1: pos(walkable)=5  holding=none  facing=EAST
    Obs A0: self_pos=4 self_ori=2 self_held=0 other_pos=5 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=5 self_ori=2 self_held=0 other_pos=4 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→W→I]                      0.005
        #3 [   W→W→S]                      0.005
        #4 [   W→W→W]                      0.005
        #5 [   I→I→I]                      0.005
    Action A0: INTERACT [5]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 9 ---
    Env state:  A0@4 held=none | A1@5 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾       🄾 
      🅿   [93m▶ [0m[38;5;208m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=none  facing=EAST
    A1: pos(walkable)=5  holding=none  facing=SOUTH
    Obs A0: self_pos=4 self_ori=2 self_held=0 other_pos=5 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=5 self_ori=1 self_held=0 other_pos=4 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→W→I]                      0.005
        #3 [   W→W→S]                      0.005
        #4 [   W→W→W]                      0.005
        #5 [   I→I→I]                      0.005
    Action A0: INTERACT [5]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 10 ---
    Env state:  A0@4 held=none | A1@2 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾     [38;5;208m▲ [0m🄾 
      🅿   [93m▶ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=none  facing=EAST
    A1: pos(walkable)=2  holding=none  facing=NORTH
    Obs A0: self_pos=4 self_ori=2 self_held=0 other_pos=2 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=2 self_ori=0 self_held=0 other_pos=4 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→W→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→S]                      0.011
        #2 [   E→I→N]                      0.011
        #3 [   E→I→S]                      0.011
        #4 [   E→I→W]                      0.011
        #5 [   E→I→E]                      0.011
    Action A0: SOUTH [1]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 11 ---
    Env state:  A0@4 held=none | A1@2 held=none | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿   [93m▼ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=none  facing=SOUTH
    A1: pos(walkable)=2  holding=none  facing=EAST
    Obs A0: self_pos=4 self_ori=1 self_held=0 other_pos=2 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=0 other_pos=4 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.198:
        #1 [   I→N→N]                      0.012
        #2 [   I→N→S]                      0.012
        #3 [   I→N→S]                      0.012
        #4 [   I→S→N]                      0.012
        #5 [   I→E→N]                      0.012
    Action A0: STAY [4]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 12 ---
    Env state:  A0@4 held=none | A1@2 held=onion | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿   [93m▼ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=none  facing=SOUTH
    A1: pos(walkable)=2  holding=onion  facing=EAST
    Obs A0: self_pos=4 self_ori=1 self_held=0 other_pos=2 other_held=1 pot=0 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=1 other_pos=4 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.349:
        #1 [   W→N→I]                      0.011
        #2 [   N→I→S]                      0.011
        #3 [   N→I→W]                      0.011
        #4 [   N→I→E]                      0.011
        #5 [   N→I→S]                      0.011
    Action A0: INTERACT [5]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 13 ---
    Env state:  A0@4 held=none | A1@1 held=onion | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾   [38;5;208m◀ [0m  🄾 
      🅿   [93m▼ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=none  facing=SOUTH
    A1: pos(walkable)=1  holding=onion  facing=WEST
    Obs A0: self_pos=4 self_ori=1 self_held=0 other_pos=1 other_held=1 pot=0 delivered=0
    Obs A1: self_pos=1 self_ori=3 self_held=1 other_pos=4 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.224:
        #1 [   N→I→S]                      0.023
        #2 [   N→I→W]                      0.023
        #3 [   N→I→E]                      0.023
        #4 [   N→I→S]                      0.023
        #5 [   N→I→N]                      0.023
    Action A0: EAST [2]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 14 ---
    Env state:  A0@5 held=none | A1@1 held=onion | pot=empty
    Map (before action):
      🅲 🅲 ⓪ 🅲 🅲 
      🄾   [38;5;208m▲ [0m  🄾 
      🅿     [93m▶ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=5  holding=none  facing=EAST
    A1: pos(walkable)=1  holding=onion  facing=NORTH
    Obs A0: self_pos=5 self_ori=2 self_held=0 other_pos=1 other_held=1 pot=0 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=1 other_pos=5 other_held=0 pot=0 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 4.661:
        #1 [   I→N→S]                      0.020
        #2 [   I→W→S]                      0.020
        #3 [   I→S→E]                      0.020
        #4 [   I→S→W]                      0.020
        #5 [   I→S→S]                      0.020
    Action A0: INTERACT [5]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 15 ---
    Env state:  A0@5 held=none | A1@1 held=none | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾   [38;5;208m▲ [0m  🄾 
      🅿     [93m▶ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=5  holding=none  facing=EAST
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=5 self_ori=2 self_held=0 other_pos=1 other_held=0 pot=1 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=5 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→N]                      0.011
        #2 [   E→I→S]                      0.011
        #3 [   E→I→E]                      0.011
        #4 [   E→I→W]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: SOUTH [1]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 16 ---
    Env state:  A0@5 held=none | A1@1 held=none | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾   [38;5;208m▲ [0m  🄾 
      🅿     [93m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=5  holding=none  facing=SOUTH
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=5 self_ori=1 self_held=0 other_pos=1 other_held=0 pot=1 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=5 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→S]                      0.011
        #2 [   E→I→W]                      0.011
        #3 [   E→I→N]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: SOUTH [1]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 17 ---
    Env state:  A0@5 held=none | A1@1 held=none | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾   [38;5;208m▲ [0m  🄾 
      🅿     [93m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=5  holding=none  facing=SOUTH
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=5 self_ori=1 self_held=0 other_pos=1 other_held=0 pot=1 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=5 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→N]                      0.011
        #2 [   E→I→S]                      0.011
        #3 [   E→I→E]                      0.011
        #4 [   E→I→W]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: INTERACT [5]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 18 ---
    Env state:  A0@5 held=none | A1@0 held=none | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾 [38;5;208m◀ [0m    🄾 
      🅿     [93m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=5  holding=none  facing=SOUTH
    A1: pos(walkable)=0  holding=none  facing=WEST
    Obs A0: self_pos=5 self_ori=1 self_held=0 other_pos=0 other_held=0 pot=1 delivered=0
    Obs A1: self_pos=0 self_ori=3 self_held=0 other_pos=5 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→W]                      0.011
        #2 [   E→I→N]                      0.011
        #3 [   E→I→S]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: INTERACT [5]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 19 ---
    Env state:  A0@5 held=none | A1@0 held=onion | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾 [38;5;208m◀ [0m    🄾 
      🅿     [93m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=5  holding=none  facing=SOUTH
    A1: pos(walkable)=0  holding=onion  facing=WEST
    Obs A0: self_pos=5 self_ori=1 self_held=0 other_pos=0 other_held=1 pot=1 delivered=0
    Obs A1: self_pos=0 self_ori=3 self_held=1 other_pos=5 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→W]                      0.011
        #2 [   E→I→S]                      0.011
        #3 [   E→I→E]                      0.011
        #4 [   E→I→N]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: INTERACT [5]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 20 ---
    Env state:  A0@5 held=none | A1@0 held=onion | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾 [38;5;208m◀ [0m    🄾 
      🅿     [93m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=5  holding=none  facing=SOUTH
    A1: pos(walkable)=0  holding=onion  facing=WEST
    Obs A0: self_pos=5 self_ori=1 self_held=0 other_pos=0 other_held=1 pot=1 delivered=0
    Obs A1: self_pos=0 self_ori=3 self_held=1 other_pos=5 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→N]                      0.011
        #2 [   E→I→W]                      0.011
        #3 [   E→I→S]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: WEST [3]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 21 ---
    Env state:  A0@4 held=none | A1@1 held=onion | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾   [38;5;208m▶ [0m  🄾 
      🅿   [93m◀ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=none  facing=WEST
    A1: pos(walkable)=1  holding=onion  facing=EAST
    Obs A0: self_pos=4 self_ori=3 self_held=0 other_pos=1 other_held=1 pot=1 delivered=0
    Obs A1: self_pos=1 self_ori=2 self_held=1 other_pos=4 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.354:
        #1 [   S→I→S]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→N]                      0.011
        #4 [   S→I→E]                      0.011
        #5 [   S→I→W]                      0.011
    Policy beliefs A1:
      entropy 5.198:
        #1 [   I→E→N]                      0.012
        #2 [   I→S→S]                      0.012
        #3 [   I→S→W]                      0.012
        #4 [   I→S→S]                      0.012
        #5 [   I→S→I]                      0.012
    Action A0: WEST [3]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 22 ---
    Env state:  A0@3 held=none | A1@1 held=onion | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾   [38;5;208m▶ [0m  🄾 
      🅿 [93m◀ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=none  facing=WEST
    A1: pos(walkable)=1  holding=onion  facing=EAST
    Obs A0: self_pos=3 self_ori=3 self_held=0 other_pos=1 other_held=1 pot=1 delivered=0
    Obs A1: self_pos=1 self_ori=2 self_held=1 other_pos=3 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.354:
        #1 [   S→I→S]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→N]                      0.011
        #4 [   S→I→E]                      0.011
        #5 [   S→I→W]                      0.011
    Policy beliefs A1:
      entropy 5.349:
        #1 [   W→N→I]                      0.011
        #2 [   N→I→W]                      0.011
        #3 [   N→I→S]                      0.011
        #4 [   N→I→S]                      0.011
        #5 [   N→I→E]                      0.011
    Action A0: EAST [2]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 23 ---
    Env state:  A0@3 held=none | A1@1 held=onion | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾   [38;5;208m▼ [0m  🄾 
      🅿 [93m▶ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=none  facing=EAST
    A1: pos(walkable)=1  holding=onion  facing=SOUTH
    Obs A0: self_pos=3 self_ori=2 self_held=0 other_pos=1 other_held=1 pot=1 delivered=0
    Obs A1: self_pos=1 self_ori=1 self_held=1 other_pos=3 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.332:
        #1 [   W→N→I]                      0.010
        #2 [   N→I→S]                      0.010
        #3 [   N→I→W]                      0.010
        #4 [   E→I→S]                      0.010
        #5 [   E→I→W]                      0.010
    Action A0: NORTH [0]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 24 ---
    Env state:  A0@0 held=none | A1@1 held=onion | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾 [93m▲ [0m[38;5;208m▼ [0m  🄾 
      🅿       🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=none  facing=NORTH
    A1: pos(walkable)=1  holding=onion  facing=SOUTH
    Obs A0: self_pos=0 self_ori=0 self_held=0 other_pos=1 other_held=1 pot=1 delivered=0
    Obs A1: self_pos=1 self_ori=1 self_held=1 other_pos=0 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.338:
        #1 [   W→I→S]                      0.010
        #2 [   W→I→S]                      0.010
        #3 [   W→I→W]                      0.010
        #4 [   W→I→E]                      0.010
        #5 [   W→I→N]                      0.010
    Policy beliefs A1:
      entropy 5.332:
        #1 [   W→N→I]                      0.010
        #2 [   E→I→S]                      0.010
        #3 [   N→I→S]                      0.010
        #4 [   N→I→W]                      0.010
        #5 [   E→I→W]                      0.010
    Action A0: WEST [3]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 25 ---
    Env state:  A0@0 held=none | A1@1 held=onion | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾 [93m◀ [0m[38;5;208m▼ [0m  🄾 
      🅿       🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=none  facing=WEST
    A1: pos(walkable)=1  holding=onion  facing=SOUTH
    Obs A0: self_pos=0 self_ori=3 self_held=0 other_pos=1 other_held=1 pot=1 delivered=0
    Obs A1: self_pos=1 self_ori=1 self_held=1 other_pos=0 other_held=0 pot=1 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.204:
        #1 [   I→S→I]                      0.012
        #2 [   I→N→N]                      0.012
        #3 [   I→N→E]                      0.012
        #4 [   I→N→S]                      0.012
        #5 [   I→E→N]                      0.012
    Policy beliefs A1:
      entropy 5.332:
        #1 [   W→N→I]                      0.010
        #2 [   E→I→S]                      0.010
        #3 [   E→I→W]                      0.010
        #4 [   N→I→S]                      0.010
        #5 [   N→I→W]                      0.010
    Action A0: INTERACT [5]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 26 ---
    Env state:  A0@0 held=onion | A1@1 held=onion | pot=1onion(idle)
    Map (before action):
      🅲 🅲 ① 🅲 🅲 
      🄾 [93m◀ [0m[38;5;208m▲ [0m  🄾 
      🅿       🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=onion  facing=WEST
    A1: pos(walkable)=1  holding=onion  facing=NORTH
    Obs A0: self_pos=0 self_ori=3 self_held=1 other_pos=1 other_held=1 pot=1 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=1 other_pos=0 other_held=1 pot=1 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.349:
        #1 [   E→N→I]                      0.011
        #2 [   N→I→S]                      0.011
        #3 [   N→I→S]                      0.011
        #4 [   N→I→N]                      0.011
        #5 [   N→I→E]                      0.011
    Policy beliefs A1:
      entropy 5.183:
        #1 [   I→W→N]                      0.012
        #2 [   I→W→S]                      0.012
        #3 [   I→W→W]                      0.012
        #4 [   I→W→S]                      0.012
        #5 [   I→W→I]                      0.012
    Action A0: WEST [3]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 27 ---
    Env state:  A0@0 held=onion | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅲 ② 🅲 🅲 
      🄾 [93m◀ [0m[38;5;208m▲ [0m  🄾 
      🅿       🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=onion  facing=WEST
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=0 self_ori=3 self_held=1 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=0 other_held=1 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=0.92, H=0.29)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.349:
        #1 [   E→N→I]                      0.011
        #2 [   N→I→S]                      0.011
        #3 [   N→I→S]                      0.011
        #4 [   N→I→N]                      0.011
        #5 [   N→I→E]                      0.011
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→S]                      0.011
        #2 [   E→I→S]                      0.011
        #3 [   E→I→W]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→N]                      0.011
    Action A0: SOUTH [1]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 28 ---
    Env state:  A0@3 held=onion | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅲 ② 🅲 🅲 
      🄾   [38;5;208m▲ [0m  🄾 
      🅿 [93m▼ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=onion  facing=SOUTH
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=3 self_ori=1 self_held=1 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=3 other_held=1 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=0.01)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.349:
        #1 [   E→N→I]                      0.011
        #2 [   N→I→S]                      0.011
        #3 [   N→I→S]                      0.011
        #4 [   N→I→N]                      0.011
        #5 [   N→I→E]                      0.011
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→S]                      0.011
        #2 [   E→I→N]                      0.011
        #3 [   E→I→W]                      0.011
        #4 [   E→I→S]                      0.011
        #5 [   E→I→E]                      0.011
    Action A0: NORTH [0]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 29 ---
    Env state:  A0@0 held=onion | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅲 ② 🅲 🅲 
      🄾 [93m▲ [0m    🄾 
      🅿   [38;5;208m▼ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=onion  facing=NORTH
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=0 self_ori=0 self_held=1 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=0 other_held=1 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.177:
        #1 [   I→S→I]                      0.017
        #2 [   I→N→S]                      0.011
        #3 [   I→S→S]                      0.011
        #4 [   I→E→S]                      0.011
        #5 [   I→E→W]                      0.011
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→I→I]                      0.005
        #3 [   W→I→S]                      0.005
        #4 [   W→I→E]                      0.005
        #5 [   W→I→S]                      0.005
    Action A0: INTERACT [5]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 30 ---
    Env state:  A0@0 held=none | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾 [93m▲ [0m    🄾 
      🅿   [38;5;208m▼ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=none  facing=NORTH
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=0 self_ori=0 self_held=0 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=0 other_held=0 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=0.01)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.338:
        #1 [   S→I→E]                      0.011
        #2 [   S→I→N]                      0.011
        #3 [   S→I→W]                      0.011
        #4 [   S→I→S]                      0.011
        #5 [   S→I→S]                      0.011
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→I→I]                      0.005
        #3 [   W→I→S]                      0.005
        #4 [   W→I→E]                      0.005
        #5 [   W→I→S]                      0.005
    Action A0: NORTH [0]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 31 ---
    Env state:  A0@0 held=none | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾 [93m▲ [0m    🄾 
      🅿   [38;5;208m▼ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=none  facing=NORTH
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=0 self_ori=0 self_held=0 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=0 other_held=0 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.338:
        #1 [   S→I→E]                      0.010
        #2 [   S→I→N]                      0.010
        #3 [   S→I→W]                      0.010
        #4 [   S→I→S]                      0.010
        #5 [   S→I→S]                      0.010
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→I→I]                      0.005
        #3 [   W→I→S]                      0.005
        #4 [   W→I→E]                      0.005
        #5 [   W→I→S]                      0.005
    Action A0: SOUTH [1]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 32 ---
    Env state:  A0@3 held=none | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿 [93m▼ [0m[38;5;208m▼ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=none  facing=SOUTH
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=3 self_ori=1 self_held=0 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=3 other_held=0 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.198:
        #1 [   I→N→I]                      0.012
        #2 [   I→E→N]                      0.012
        #3 [   I→E→E]                      0.012
        #4 [   I→E→S]                      0.012
        #5 [   I→E→I]                      0.012
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→I→I]                      0.005
        #3 [   W→I→S]                      0.005
        #4 [   W→I→E]                      0.005
        #5 [   W→I→S]                      0.005
    Action A0: INTERACT [5]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 33 ---
    Env state:  A0@3 held=dish | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿 [93m▼ [0m[38;5;208m▼ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   I→I→I]                      0.005
        #3 [   S→S→S]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→I→I]                      0.005
        #3 [   W→I→S]                      0.005
        #4 [   W→I→E]                      0.005
        #5 [   W→I→S]                      0.005
    Action A0: SOUTH [1]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 34 ---
    Env state:  A0@3 held=dish | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿 [93m▼ [0m[38;5;208m◀ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=4  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   I→I→I]                      0.005
        #3 [   S→S→S]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   I→I→I]                      0.005
        #4 [   S→I→N]                      0.005
        #5 [   I→N→W]                      0.005
    Action A0: WEST [3]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 35 ---
    Env state:  A0@3 held=dish | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿 [93m◀ [0m[38;5;208m◀ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=4  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   I→I→I]                      0.005
        #4 [   S→I→N]                      0.005
        #5 [   I→N→W]                      0.005
    Action A0: WEST [3]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 36 ---
    Env state:  A0@3 held=dish | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿 [93m◀ [0m[38;5;208m◀ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=4  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: INTERACT [5]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 37 ---
    Env state:  A0@3 held=dish | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿 [93m◀ [0m[38;5;208m◀ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=4  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: WEST [3]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 38 ---
    Env state:  A0@3 held=dish | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿 [93m◀ [0m[38;5;208m◀ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=4  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: NORTH [0]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 39 ---
    Env state:  A0@0 held=dish | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾 [93m▲ [0m    🄾 
      🅿   [38;5;208m◀ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=NORTH
    A1: pos(walkable)=4  holding=none  facing=WEST
    Obs A0: self_pos=0 self_ori=0 self_held=2 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=3 self_held=0 other_pos=0 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→S→S]                      0.005
        #3 [   S→S→S]                      0.005
        #4 [   S→I→I]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: STAY [4]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 40 ---
    Env state:  A0@0 held=dish | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾 [93m▲ [0m    🄾 
      🅿   [38;5;208m◀ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=NORTH
    A1: pos(walkable)=4  holding=none  facing=WEST
    Obs A0: self_pos=0 self_ori=0 self_held=2 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=3 self_held=0 other_pos=0 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→S→S]                      0.005
        #3 [   S→S→S]                      0.005
        #4 [   S→I→I]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: WEST [3]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 41 ---
    Env state:  A0@0 held=dish | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾 [93m◀ [0m[38;5;208m▲ [0m  🄾 
      🅿       🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=WEST
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=0 self_ori=3 self_held=2 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=0 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: STAY [4]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 42 ---
    Env state:  A0@0 held=dish | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾 [93m◀ [0m[38;5;208m◀ [0m  🄾 
      🅿       🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=WEST
    A1: pos(walkable)=1  holding=none  facing=WEST
    Obs A0: self_pos=0 self_ori=3 self_held=2 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=3 self_held=0 other_pos=0 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: INTERACT [5]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 43 ---
    Env state:  A0@0 held=dish | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾 [93m◀ [0m[38;5;208m▲ [0m  🄾 
      🅿       🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=WEST
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=0 self_ori=3 self_held=2 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=0 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: SOUTH [1]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 44 ---
    Env state:  A0@3 held=dish | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾   [38;5;208m▲ [0m  🄾 
      🅿 [93m▼ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   I→I→I]                      0.005
        #3 [   S→S→S]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: NORTH [0]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 45 ---
    Env state:  A0@3 held=dish | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾   [38;5;208m◀ [0m  🄾 
      🅿 [93m▲ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=NORTH
    A1: pos(walkable)=1  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=0 self_held=2 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→S→S]                      0.005
        #3 [   S→S→S]                      0.005
        #4 [   S→I→I]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: SOUTH [1]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 46 ---
    Env state:  A0@3 held=dish | A1@2 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿 [93m▼ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=2  holding=none  facing=EAST
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=2 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   I→I→I]                      0.005
        #3 [   S→S→S]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.374:
        #1 [   N→E→I]                      0.007
        #2 [   W→S→I]                      0.007
        #3 [   N→W→I]                      0.005
        #4 [   E→W→S]                      0.005
        #5 [   E→W→I]                      0.005
    Action A0: WEST [3]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 47 ---
    Env state:  A0@3 held=dish | A1@2 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿 [93m◀ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=2  holding=onion  facing=EAST
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=2 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=1 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.374:
        #1 [   N→E→I]                      0.007
        #2 [   W→S→I]                      0.007
        #3 [   N→W→I]                      0.005
        #4 [   E→W→S]                      0.005
        #5 [   E→W→I]                      0.005
    Action A0: WEST [3]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 48 ---
    Env state:  A0@3 held=dish | A1@2 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿 [93m◀ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=2  holding=onion  facing=EAST
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=2 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=1 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.374:
        #1 [   N→E→I]                      0.007
        #2 [   W→S→I]                      0.007
        #3 [   N→W→I]                      0.005
        #4 [   E→W→S]                      0.005
        #5 [   E→W→I]                      0.005
    Action A0: WEST [3]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 49 ---
    Env state:  A0@3 held=dish | A1@2 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿 [93m◀ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=2  holding=onion  facing=EAST
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=2 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=1 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→I→N]                      0.005
        #3 [   W→I→S]                      0.005
        #4 [   W→I→I]                      0.005
        #5 [   S→W→I]                      0.005
    Action A0: WEST [3]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 50 ---
    Env state:  A0@3 held=dish | A1@2 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▲ [0m🄾 
      🅿 [93m◀ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=2  holding=onion  facing=NORTH
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=2 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=0 self_held=1 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→N]                      0.011
        #2 [   E→I→S]                      0.011
        #3 [   E→I→W]                      0.011
        #4 [   E→I→S]                      0.011
        #5 [   E→I→E]                      0.011
    Action A0: EAST [2]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 51 ---
    Env state:  A0@4 held=dish | A1@2 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿   [93m▶ [0m  🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=dish  facing=EAST
    A1: pos(walkable)=2  holding=onion  facing=EAST
    Obs A0: self_pos=4 self_ori=2 self_held=2 other_pos=2 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=1 other_pos=4 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.352:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→E]                      0.011
        #3 [   S→I→W]                      0.011
        #4 [   S→I→S]                      0.011
        #5 [   S→I→S]                      0.011
    Policy beliefs A1:
      entropy 5.198:
        #1 [   I→N→N]                      0.012
        #2 [   I→N→S]                      0.012
        #3 [   I→E→N]                      0.012
        #4 [   I→W→S]                      0.012
        #5 [   I→S→S]                      0.012
    Action A0: WEST [3]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 52 ---
    Env state:  A0@3 held=dish | A1@2 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿 [93m◀ [0m    🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=2  holding=onion  facing=EAST
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=2 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=1 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→S→I]                      0.007
        #2 [   S→I→S]                      0.005
        #3 [   S→I→N]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.349:
        #1 [   W→N→I]                      0.011
        #2 [   N→I→W]                      0.011
        #3 [   N→I→S]                      0.011
        #4 [   N→I→E]                      0.011
        #5 [   N→I→S]                      0.011
    Action A0: EAST [2]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 53 ---
    Env state:  A0@4 held=dish | A1@5 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿   [93m▶ [0m[38;5;208m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=dish  facing=EAST
    A1: pos(walkable)=5  holding=onion  facing=SOUTH
    Obs A0: self_pos=4 self_ori=2 self_held=2 other_pos=5 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=5 self_ori=1 self_held=1 other_pos=4 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.352:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→E]                      0.011
        #3 [   S→I→W]                      0.011
        #4 [   S→I→S]                      0.011
        #5 [   S→I→S]                      0.011
    Policy beliefs A1:
      entropy 5.332:
        #1 [   W→N→I]                      0.010
        #2 [   E→I→S]                      0.010
        #3 [   N→I→S]                      0.010
        #4 [   N→I→W]                      0.010
        #5 [   E→I→E]                      0.010
    Action A0: INTERACT [5]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 54 ---
    Env state:  A0@4 held=dish | A1@5 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿   [93m▶ [0m[38;5;208m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=dish  facing=EAST
    A1: pos(walkable)=5  holding=onion  facing=SOUTH
    Obs A0: self_pos=4 self_ori=2 self_held=2 other_pos=5 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=5 self_ori=1 self_held=1 other_pos=4 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.352:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→E]                      0.011
        #3 [   S→I→W]                      0.011
        #4 [   S→I→S]                      0.011
        #5 [   S→I→S]                      0.011
    Policy beliefs A1:
      entropy 5.332:
        #1 [   W→N→I]                      0.010
        #2 [   E→I→N]                      0.010
        #3 [   E→I→S]                      0.010
        #4 [   E→I→E]                      0.010
        #5 [   N→I→N]                      0.010
    Action A0: STAY [4]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 55 ---
    Env state:  A0@4 held=dish | A1@5 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿   [93m▶ [0m[38;5;208m▼ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=dish  facing=EAST
    A1: pos(walkable)=5  holding=onion  facing=SOUTH
    Obs A0: self_pos=4 self_ori=2 self_held=2 other_pos=5 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=5 self_ori=1 self_held=1 other_pos=4 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.352:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→E]                      0.011
        #3 [   S→I→W]                      0.011
        #4 [   S→I→S]                      0.011
        #5 [   S→I→S]                      0.011
    Policy beliefs A1:
      entropy 5.332:
        #1 [   W→N→I]                      0.010
        #2 [   E→I→N]                      0.010
        #3 [   E→I→S]                      0.010
        #4 [   N→I→S]                      0.010
        #5 [   N→I→E]                      0.010
    Action A0: INTERACT [5]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 56 ---
    Env state:  A0@4 held=dish | A1@5 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿   [93m▶ [0m[38;5;208m▶ [0m🅲 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=dish  facing=EAST
    A1: pos(walkable)=5  holding=onion  facing=EAST
    Obs A0: self_pos=4 self_ori=2 self_held=2 other_pos=5 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=5 self_ori=2 self_held=1 other_pos=4 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.352:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→E]                      0.011
        #3 [   S→I→W]                      0.011
        #4 [   S→I→S]                      0.011
        #5 [   S→I→S]                      0.011
    Policy beliefs A1:
      entropy 5.192:
        #1 [   I→W→S]                      0.011
        #2 [   I→W→W]                      0.011
        #3 [   I→W→S]                      0.011
        #4 [   I→W→I]                      0.011
        #5 [   I→W→N]                      0.011
    Action A0: INTERACT [5]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 57 ---
    Env state:  A0@4 held=dish | A1@5 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿   [93m▶ [0m[38;5;208m▶ [0m🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=dish  facing=EAST
    A1: pos(walkable)=5  holding=none  facing=EAST
    Obs A0: self_pos=4 self_ori=2 self_held=2 other_pos=5 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=5 self_ori=2 self_held=0 other_pos=4 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=0.01)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.352:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→E]                      0.011
        #3 [   S→I→W]                      0.011
        #4 [   S→I→S]                      0.011
        #5 [   S→I→S]                      0.011
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→I→N]                      0.005
        #3 [   W→I→S]                      0.005
        #4 [   W→I→I]                      0.005
        #5 [   S→W→I]                      0.005
    Action A0: SOUTH [1]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 58 ---
    Env state:  A0@4 held=dish | A1@5 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾       🄾 
      🅿   [93m▼ [0m[38;5;208m▼ [0m🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=dish  facing=SOUTH
    A1: pos(walkable)=5  holding=none  facing=SOUTH
    Obs A0: self_pos=4 self_ori=1 self_held=2 other_pos=5 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=5 self_ori=1 self_held=0 other_pos=4 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.181:
        #1 [   I→E→S]                      0.012
        #2 [   I→W→S]                      0.012
        #3 [   I→N→E]                      0.012
        #4 [   I→N→W]                      0.012
        #5 [   I→N→S]                      0.012
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→I→I]                      0.005
        #3 [   W→I→S]                      0.005
        #4 [   W→I→E]                      0.005
        #5 [   W→I→S]                      0.005
    Action A0: INTERACT [5]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 59 ---
    Env state:  A0@4 held=none | A1@2 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▲ [0m🄾 
      🅿   [93m▼ [0m  🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=4  holding=none  facing=SOUTH
    A1: pos(walkable)=2  holding=none  facing=NORTH
    Obs A0: self_pos=4 self_ori=1 self_held=0 other_pos=2 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=0 self_held=0 other_pos=4 other_held=0 pot=2 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=0.01)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   N→E→I]                      0.007
        #2 [   N→W→I]                      0.007
        #3 [   W→S→I]                      0.007
        #4 [   I→I→I]                      0.005
        #5 [   E→S→E]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→N]                      0.011
        #2 [   E→I→W]                      0.011
        #3 [   E→I→S]                      0.011
        #4 [   E→I→S]                      0.011
        #5 [   E→I→E]                      0.011
    Action A0: WEST [3]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 60 ---
    Env state:  A0@3 held=none | A1@2 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿 [93m◀ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=none  facing=WEST
    A1: pos(walkable)=2  holding=none  facing=EAST
    Obs A0: self_pos=3 self_ori=3 self_held=0 other_pos=2 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=0 other_pos=3 other_held=0 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Policy beliefs A1:
      entropy 5.198:
        #1 [   I→W→S]                      0.012
        #2 [   I→N→N]                      0.012
        #3 [   I→N→W]                      0.012
        #4 [   I→N→S]                      0.012
        #5 [   I→W→N]                      0.012
    Action A0: SOUTH [1]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 61 ---
    Env state:  A0@3 held=none | A1@2 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿 [93m▼ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=none  facing=SOUTH
    A1: pos(walkable)=2  holding=onion  facing=EAST
    Obs A0: self_pos=3 self_ori=1 self_held=0 other_pos=2 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=1 other_pos=3 other_held=0 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.198:
        #1 [   I→S→W]                      0.012
        #2 [   I→S→W]                      0.012
        #3 [   I→W→N]                      0.012
        #4 [   I→W→S]                      0.012
        #5 [   I→W→E]                      0.012
    Policy beliefs A1:
      entropy 5.349:
        #1 [   W→N→I]                      0.011
        #2 [   N→I→S]                      0.011
        #3 [   N→I→S]                      0.011
        #4 [   N→I→N]                      0.011
        #5 [   N→I→E]                      0.011
    Action A0: INTERACT [5]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 62 ---
    Env state:  A0@3 held=dish | A1@2 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅲 🅲 
      🄾     [38;5;208m▲ [0m🄾 
      🅿 [93m▼ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=2  holding=onion  facing=NORTH
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=2 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=0 self_held=1 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   I→I→I]                      0.005
        #2 [   S→S→N]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.182:
        #1 [   I→S→W]                      0.012
        #2 [   I→W→I]                      0.012
        #3 [   I→W→S]                      0.012
        #4 [   I→W→W]                      0.012
        #5 [   I→W→S]                      0.012
    Action A0: SOUTH [1]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 63 ---
    Env state:  A0@3 held=dish | A1@2 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾     [38;5;208m▲ [0m🄾 
      🅿 [93m▼ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=2  holding=none  facing=NORTH
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=2 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=0 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.01)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   I→I→I]                      0.005
        #2 [   S→S→N]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→N]                      0.011
        #2 [   E→I→S]                      0.011
        #3 [   E→I→W]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: STAY [4]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 64 ---
    Env state:  A0@3 held=dish | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾   [38;5;208m◀ [0m  🄾 
      🅿 [93m▼ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=1  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   I→I→I]                      0.005
        #2 [   S→S→N]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→W]                      0.011
        #2 [   E→I→N]                      0.011
        #3 [   E→I→S]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: STAY [4]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 65 ---
    Env state:  A0@3 held=dish | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾   [38;5;208m◀ [0m  🄾 
      🅿 [93m▼ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=1  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   I→I→I]                      0.005
        #2 [   S→S→N]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→W]                      0.011
        #2 [   E→I→N]                      0.011
        #3 [   E→I→S]                      0.011
        #4 [   E→I→S]                      0.011
        #5 [   E→I→E]                      0.011
    Action A0: WEST [3]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 66 ---
    Env state:  A0@3 held=dish | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾   [38;5;208m▲ [0m  🄾 
      🅿 [93m◀ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   S→I→S]                      0.005
        #2 [   S→I→N]                      0.005
        #3 [   S→I→S]                      0.005
        #4 [   S→N→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→N]                      0.011
        #2 [   E→I→S]                      0.011
        #3 [   E→I→E]                      0.011
        #4 [   E→I→W]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: NORTH [0]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 67 ---
    Env state:  A0@3 held=dish | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾   [38;5;208m◀ [0m  🄾 
      🅿 [93m▲ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=NORTH
    A1: pos(walkable)=1  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=0 self_held=2 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   S→S→S]                      0.005
        #2 [   S→S→I]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   I→S→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→W]                      0.011
        #2 [   E→I→N]                      0.011
        #3 [   E→I→S]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: SOUTH [1]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 68 ---
    Env state:  A0@3 held=dish | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾       🄾 
      🅿 [93m▼ [0m[38;5;208m▼ [0m  🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   I→I→I]                      0.005
        #2 [   S→S→N]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   N→W→I]                      0.005
        #4 [   E→W→S]                      0.005
        #5 [   E→W→I]                      0.005
    Action A0: INTERACT [5]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 69 ---
    Env state:  A0@3 held=dish | A1@5 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾       🄾 
      🅿 [93m▼ [0m  [38;5;208m▶ [0m🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=5  holding=none  facing=EAST
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=5 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=5 self_ori=2 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   I→I→I]                      0.005
        #2 [   S→S→N]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.375:
        #1 [   N→E→I]                      0.007
        #2 [   W→I→N]                      0.005
        #3 [   W→I→S]                      0.005
        #4 [   W→I→I]                      0.005
        #5 [   S→W→I]                      0.005
    Action A0: NORTH [0]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 70 ---
    Env state:  A0@0 held=dish | A1@4 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾 [93m▲ [0m    🄾 
      🅿   [38;5;208m◀ [0m  🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=NORTH
    A1: pos(walkable)=4  holding=none  facing=WEST
    Obs A0: self_pos=0 self_ori=0 self_held=2 other_pos=4 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=4 self_ori=3 self_held=0 other_pos=0 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   S→S→S]                      0.005
        #2 [   S→S→I]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   I→S→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   I→I→I]                      0.005
        #4 [   S→I→N]                      0.005
        #5 [   I→N→W]                      0.005
    Action A0: SOUTH [1]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 71 ---
    Env state:  A0@3 held=dish | A1@1 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾   [38;5;208m▲ [0m  🄾 
      🅿 [93m▼ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=1 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   I→I→I]                      0.005
        #2 [   S→S→N]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→N]                      0.011
        #2 [   E→I→S]                      0.011
        #3 [   E→I→W]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: INTERACT [5]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 72 ---
    Env state:  A0@3 held=dish | A1@2 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿 [93m▼ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=2  holding=none  facing=EAST
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=2 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=0 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   I→I→I]                      0.005
        #2 [   S→S→N]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   S→I→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.198:
        #1 [   I→N→N]                      0.012
        #2 [   I→N→S]                      0.012
        #3 [   I→N→I]                      0.012
        #4 [   I→E→N]                      0.012
        #5 [   I→W→W]                      0.012
    Action A0: EAST [2]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 73 ---
    Env state:  A0@4 held=dish | A1@2 held=none | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿   [93m▶ [0m  🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=4  holding=dish  facing=EAST
    A1: pos(walkable)=2  holding=none  facing=EAST
    Obs A0: self_pos=4 self_ori=2 self_held=2 other_pos=2 other_held=0 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=0 other_pos=4 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   N→N→N]                      0.005
        #2 [   N→E→S]                      0.005
        #3 [   S→N→I]                      0.005
        #4 [   S→N→S]                      0.005
        #5 [   S→N→E]                      0.005
    Policy beliefs A1:
      entropy 5.198:
        #1 [   I→N→N]                      0.012
        #2 [   I→N→S]                      0.012
        #3 [   I→N→I]                      0.012
        #4 [   I→E→N]                      0.012
        #5 [   I→W→W]                      0.012
    Action A0: WEST [3]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 74 ---
    Env state:  A0@3 held=dish | A1@2 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾     [38;5;208m▶ [0m🄾 
      🅿 [93m◀ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=WEST
    A1: pos(walkable)=2  holding=onion  facing=EAST
    Obs A0: self_pos=3 self_ori=3 self_held=2 other_pos=2 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=1 other_pos=3 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   S→I→S]                      0.005
        #2 [   S→I→N]                      0.005
        #3 [   S→I→S]                      0.005
        #4 [   S→N→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.371:
        #1 [   W→N→I]                      0.012
        #2 [   W→S→I]                      0.007
        #3 [   W→W→S]                      0.005
        #4 [   S→S→S]                      0.005
        #5 [   W→W→I]                      0.005
    Action A0: NORTH [0]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 75 ---
    Env state:  A0@0 held=dish | A1@1 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾 [93m▲ [0m[38;5;208m◀ [0m  🄾 
      🅿       🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=NORTH
    A1: pos(walkable)=1  holding=onion  facing=WEST
    Obs A0: self_pos=0 self_ori=0 self_held=2 other_pos=1 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=3 self_held=1 other_pos=0 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   S→S→S]                      0.005
        #2 [   S→S→I]                      0.005
        #3 [   S→I→I]                      0.005
        #4 [   I→S→S]                      0.005
        #5 [   S→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.223:
        #1 [   N→I→N]                      0.023
        #2 [   N→I→S]                      0.023
        #3 [   N→I→E]                      0.023
        #4 [   N→I→W]                      0.023
        #5 [   N→I→S]                      0.023
    Action A0: WEST [3]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 76 ---
    Env state:  A0@0 held=dish | A1@1 held=onion | pot=2onion(idle)
    Map (before action):
      🅲 🅾 ② 🅾 🅲 
      🄾 [93m◀ [0m[38;5;208m▲ [0m  🄾 
      🅿       🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=WEST
    A1: pos(walkable)=1  holding=onion  facing=NORTH
    Obs A0: self_pos=0 self_ori=3 self_held=2 other_pos=1 other_held=1 pot=2 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=1 other_pos=0 other_held=2 pot=2 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=1.00, H=0.00)
      pot_state 1 (p=1.00, H=-0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   S→I→S]                      0.005
        #2 [   S→I→N]                      0.005
        #3 [   S→I→S]                      0.005
        #4 [   S→N→S]                      0.005
        #5 [   S→S→N]                      0.005
    Policy beliefs A1:
      entropy 4.658:
        #1 [   I→W→W]                      0.020
        #2 [   I→W→S]                      0.020
        #3 [   I→N→N]                      0.020
        #4 [   I→N→S]                      0.020
        #5 [   I→W→N]                      0.020
    Action A0: EAST [2]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 77 ---
    Env state:  A0@0 held=dish | A1@1 held=none | pot=3onion(ready t=0/0)
    Map (before action):
      🅲 🅾 ❸ 🅾 🅲 
      🄾 [93m▶ [0m[38;5;208m▲ [0m  🄾 
      🅿       🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=EAST
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=0 self_ori=2 self_held=2 other_pos=1 other_held=0 pot=3 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=0 other_held=2 pot=3 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 2 (p=0.90, H=0.33)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→N→I]                      0.007
        #2 [   E→I→I]                      0.005
        #3 [   E→N→E]                      0.005
        #4 [   E→S→E]                      0.005
        #5 [   E→S→W]                      0.005
    Policy beliefs A1:
      entropy 5.355:
        #1 [   E→I→W]                      0.011
        #2 [   E→I→N]                      0.011
        #3 [   E→I→S]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: SOUTH [1]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 78 ---
    Env state:  A0@3 held=dish | A1@1 held=none | pot=3onion(ready t=0/0)
    Map (before action):
      🅲 🅾 ❸ 🅾 🅲 
      🄾   [38;5;208m▲ [0m  🄾 
      🅿 [93m▼ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=SOUTH
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=3 self_ori=1 self_held=2 other_pos=1 other_held=0 pot=3 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=3 other_held=2 pot=3 delivered=0
    Beliefs A0:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 2 (p=0.83, H=0.58)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=0.90, H=0.33)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   N→N→N]                      0.005
        #2 [   N→I→E]                      0.005
        #3 [   N→S→E]                      0.005
        #4 [   N→S→S]                      0.005
        #5 [   I→N→I]                      0.005
    Policy beliefs A1:
      entropy 5.355:
        #1 [   E→I→W]                      0.011
        #2 [   E→I→N]                      0.011
        #3 [   E→I→S]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: NORTH [0]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 79 ---
    Env state:  A0@3 held=dish | A1@1 held=none | pot=3onion(ready t=0/0)
    Map (before action):
      🅲 🅾 ❸ 🅾 🅲 
      🄾   [38;5;208m◀ [0m  🄾 
      🅿 [93m▲ [0m    🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=3  holding=dish  facing=NORTH
    A1: pos(walkable)=1  holding=none  facing=WEST
    Obs A0: self_pos=3 self_ori=0 self_held=2 other_pos=1 other_held=0 pot=3 delivered=0
    Obs A1: self_pos=1 self_ori=3 self_held=0 other_pos=3 other_held=2 pot=3 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=0.96, H=0.17)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=0.99, H=0.05)
      ck_put3 0 (p=0.92, H=0.28)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→N→I]                      0.007
        #2 [   E→S→W]                      0.005
        #3 [   E→E→N]                      0.005
        #4 [   E→E→S]                      0.005
        #5 [   E→E→E]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→N]                      0.011
        #2 [   E→I→W]                      0.011
        #3 [   E→I→E]                      0.011
        #4 [   E→I→S]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: NORTH [0]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 80 ---
    Env state:  A0@0 held=dish | A1@1 held=none | pot=3onion(ready t=0/0)
    Map (before action):
      🅲 🅾 ❸ 🅾 🅲 
      🄾 [93m▲ [0m[38;5;208m▲ [0m  🄾 
      🅿       🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=NORTH
    A1: pos(walkable)=1  holding=none  facing=NORTH
    Obs A0: self_pos=0 self_ori=0 self_held=2 other_pos=1 other_held=0 pot=3 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=0 other_pos=0 other_held=2 pot=3 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=0.00)
      ck_put3 1 (p=0.97, H=0.15)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→N→I]                      0.007
        #2 [   E→S→W]                      0.005
        #3 [   E→E→N]                      0.005
        #4 [   E→E→S]                      0.005
        #5 [   E→E→E]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→W]                      0.011
        #2 [   E→I→N]                      0.011
        #3 [   E→I→S]                      0.011
        #4 [   E→I→E]                      0.011
        #5 [   E→I→S]                      0.011
    Action A0: INTERACT [5]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 81 ---
    Env state:  A0@0 held=dish | A1@4 held=none | pot=3onion(ready t=0/0)
    Map (before action):
      🅲 🅾 ❸ 🅾 🅲 
      🄾 [93m▲ [0m    🄾 
      🅿   [38;5;208m▼ [0m  🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=NORTH
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=0 self_ori=0 self_held=2 other_pos=4 other_held=0 pot=3 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=0 other_held=2 pot=3 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=0.00)
      ck_put3 1 (p=1.00, H=0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→N→I]                      0.007
        #2 [   E→S→W]                      0.005
        #3 [   E→E→N]                      0.005
        #4 [   E→E→S]                      0.005
        #5 [   E→E→E]                      0.005
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: NORTH [0]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 82 ---
    Env state:  A0@0 held=dish | A1@4 held=dish | pot=3onion(ready t=0/0)
    Map (before action):
      🅲 🅾 ❸ 🅾 🅲 
      🄾 [93m▲ [0m    🄾 
      🅿   [38;5;208m▼ [0m  🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=dish  facing=NORTH
    A1: pos(walkable)=4  holding=dish  facing=SOUTH
    Obs A0: self_pos=0 self_ori=0 self_held=2 other_pos=4 other_held=2 pot=3 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=2 other_pos=0 other_held=2 pot=3 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   E→N→I]                      0.007
        #2 [   E→S→W]                      0.005
        #3 [   E→E→N]                      0.005
        #4 [   E→E→S]                      0.005
        #5 [   E→E→E]                      0.005
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: EAST [2]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 83 ---
    Env state:  A0@1 held=dish | A1@4 held=dish | pot=3onion(ready t=0/0)
    Map (before action):
      🅲 🅾 ❸ 🅾 🅲 
      🄾   [93m▶ [0m  🄾 
      🅿   [38;5;208m▼ [0m  🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=1  holding=dish  facing=EAST
    A1: pos(walkable)=4  holding=dish  facing=SOUTH
    Obs A0: self_pos=1 self_ori=2 self_held=2 other_pos=4 other_held=2 pot=3 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=2 other_pos=1 other_held=2 pot=3 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.352:
        #1 [   N→I→S]                      0.011
        #2 [   N→I→S]                      0.011
        #3 [   N→I→N]                      0.011
        #4 [   N→I→W]                      0.011
        #5 [   N→I→E]                      0.011
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: NORTH [0]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 84 ---
    Env state:  A0@1 held=dish | A1@4 held=dish | pot=3onion(ready t=0/0)
    Map (before action):
      🅲 🅾 ❸ 🅾 🅲 
      🄾   [93m▲ [0m  🄾 
      🅿   [38;5;208m▲ [0m  🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=1  holding=dish  facing=NORTH
    A1: pos(walkable)=4  holding=dish  facing=NORTH
    Obs A0: self_pos=1 self_ori=0 self_held=2 other_pos=4 other_held=2 pot=3 delivered=0
    Obs A1: self_pos=4 self_ori=0 self_held=2 other_pos=1 other_held=2 pot=3 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 2 (p=1.00, H=-0.00)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 0 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.175:
        #1 [   I→E→I]                      0.017
        #2 [   I→S→S]                      0.012
        #3 [   I→S→W]                      0.012
        #4 [   I→S→S]                      0.012
        #5 [   I→S→I]                      0.012
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→W]                      0.011
        #2 [   E→I→S]                      0.011
        #3 [   E→I→N]                      0.011
        #4 [   E→I→S]                      0.011
        #5 [   E→I→E]                      0.011
    Action A0: INTERACT [5]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 85 ---
    Env state:  A0@1 held=soup | A1@4 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾   [93m▲ [0m  🄾 
      🅿   [38;5;208m▲ [0m  🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=1  holding=soup  facing=NORTH
    A1: pos(walkable)=4  holding=dish  facing=NORTH
    Obs A0: self_pos=1 self_ori=0 self_held=3 other_pos=4 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=0 self_held=2 other_pos=1 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=0.90, H=0.33)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.01)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.359:
        #1 [   E→I→S]                      0.010
        #2 [   E→I→W]                      0.010
        #3 [   E→I→N]                      0.010
        #4 [   E→I→E]                      0.010
        #5 [   E→I→S]                      0.010
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→W]                      0.011
        #2 [   E→I→S]                      0.011
        #3 [   E→I→N]                      0.011
        #4 [   E→I→S]                      0.011
        #5 [   E→I→E]                      0.011
    Action A0: EAST [2]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 86 ---
    Env state:  A0@2 held=soup | A1@4 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾     [93m▶ [0m🄾 
      🅿   [38;5;208m▼ [0m  🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=2  holding=soup  facing=EAST
    A1: pos(walkable)=4  holding=dish  facing=SOUTH
    Obs A0: self_pos=2 self_ori=2 self_held=3 other_pos=4 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=2 other_pos=2 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=0.90, H=0.33)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.237:
        #1 [   I→N→I]                      0.011
        #2 [   I→S→S]                      0.011
        #3 [   I→S→W]                      0.011
        #4 [   I→S→S]                      0.011
        #5 [   I→S→I]                      0.011
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: NORTH [0]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 87 ---
    Env state:  A0@2 held=soup | A1@4 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾     [93m▲ [0m🄾 
      🅿   [38;5;208m▼ [0m  🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=2  holding=soup  facing=NORTH
    A1: pos(walkable)=4  holding=dish  facing=SOUTH
    Obs A0: self_pos=2 self_ori=0 self_held=3 other_pos=4 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=2 other_pos=2 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=0.90, H=0.33)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.359:
        #1 [   E→I→S]                      0.010
        #2 [   E→I→W]                      0.010
        #3 [   E→I→N]                      0.010
        #4 [   E→I→E]                      0.010
        #5 [   E→I→S]                      0.010
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: EAST [2]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 88 ---
    Env state:  A0@2 held=soup | A1@4 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾     [93m▶ [0m🄾 
      🅿   [38;5;208m▼ [0m  🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=2  holding=soup  facing=EAST
    A1: pos(walkable)=4  holding=dish  facing=SOUTH
    Obs A0: self_pos=2 self_ori=2 self_held=3 other_pos=4 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=2 other_pos=2 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=0.90, H=0.33)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.237:
        #1 [   I→N→I]                      0.011
        #2 [   I→S→S]                      0.011
        #3 [   I→S→W]                      0.011
        #4 [   I→S→S]                      0.011
        #5 [   I→S→I]                      0.011
    Policy beliefs A1:
      entropy 5.374:
        #1 [   W→S→I]                      0.007
        #2 [   N→E→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: STAY [4]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 89 ---
    Env state:  A0@2 held=soup | A1@4 held=none | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾     [93m▶ [0m🄾 
      🅿   [38;5;208m▼ [0m  🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=2  holding=soup  facing=EAST
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=2 self_ori=2 self_held=3 other_pos=4 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=2 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=0.90, H=0.33)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=0.00)
    Policy beliefs A0:
      entropy 5.237:
        #1 [   I→N→I]                      0.011
        #2 [   I→S→S]                      0.011
        #3 [   I→S→W]                      0.011
        #4 [   I→S→S]                      0.011
        #5 [   I→S→I]                      0.011
    Policy beliefs A1:
      entropy 5.374:
        #1 [   N→E→I]                      0.007
        #2 [   W→S→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: NORTH [0]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 90 ---
    Env state:  A0@2 held=soup | A1@4 held=none | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾     [93m▲ [0m🄾 
      🅿   [38;5;208m▼ [0m  🅾 
      🅲 🄳 🅿 🅂 🅲 
    A0: pos(walkable)=2  holding=soup  facing=NORTH
    A1: pos(walkable)=4  holding=none  facing=SOUTH
    Obs A0: self_pos=2 self_ori=0 self_held=3 other_pos=4 other_held=0 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=0 other_pos=2 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=0.90, H=0.33)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=0.00)
    Policy beliefs A0:
      entropy 5.359:
        #1 [   E→I→S]                      0.010
        #2 [   E→I→W]                      0.010
        #3 [   E→I→N]                      0.010
        #4 [   E→I→E]                      0.010
        #5 [   E→I→S]                      0.010
    Policy beliefs A1:
      entropy 5.374:
        #1 [   N→E→I]                      0.007
        #2 [   W→S→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: STAY [4]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 91 ---
    Env state:  A0@2 held=soup | A1@4 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾     [93m▲ [0m🄾 
      🅿   [38;5;208m▼ [0m  🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=2  holding=soup  facing=NORTH
    A1: pos(walkable)=4  holding=dish  facing=SOUTH
    Obs A0: self_pos=2 self_ori=0 self_held=3 other_pos=4 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=2 other_pos=2 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=0.90, H=0.33)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=0.00)
    Policy beliefs A0:
      entropy 5.359:
        #1 [   E→I→S]                      0.010
        #2 [   E→I→W]                      0.010
        #3 [   E→I→N]                      0.010
        #4 [   E→I→E]                      0.010
        #5 [   E→I→S]                      0.010
    Policy beliefs A1:
      entropy 5.374:
        #1 [   N→E→I]                      0.007
        #2 [   W→S→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: EAST [2]
    Action A1: STAY [4]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 92 ---
    Env state:  A0@2 held=soup | A1@4 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾     [93m▶ [0m🄾 
      🅿   [38;5;208m▼ [0m  🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=2  holding=soup  facing=EAST
    A1: pos(walkable)=4  holding=dish  facing=SOUTH
    Obs A0: self_pos=2 self_ori=2 self_held=3 other_pos=4 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=4 self_ori=1 self_held=2 other_pos=2 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=0.90, H=0.33)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=0.00)
    Policy beliefs A0:
      entropy 5.237:
        #1 [   I→W→S]                      0.011
        #2 [   I→S→W]                      0.011
        #3 [   I→S→S]                      0.011
        #4 [   I→S→I]                      0.011
        #5 [   I→W→N]                      0.011
    Policy beliefs A1:
      entropy 5.374:
        #1 [   N→E→I]                      0.007
        #2 [   W→S→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: INTERACT [5]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 93 ---
    Env state:  A0@2 held=soup | A1@3 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾     [93m▶ [0m🄾 
      🅿 [38;5;208m◀ [0m    🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=2  holding=soup  facing=EAST
    A1: pos(walkable)=3  holding=dish  facing=WEST
    Obs A0: self_pos=2 self_ori=2 self_held=3 other_pos=3 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=3 self_ori=3 self_held=2 other_pos=2 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=0.81, H=0.62)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=0.00)
    Policy beliefs A0:
      entropy 5.316:
        #1 [   W→N→I]                      0.012
        #2 [   W→I→S]                      0.005
        #3 [   W→S→N]                      0.005
        #4 [   W→I→N]                      0.005
        #5 [   W→S→I]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: STAY [4]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 94 ---
    Env state:  A0@2 held=soup | A1@0 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾 [38;5;208m▲ [0m  [93m▶ [0m🄾 
      🅿       🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=2  holding=soup  facing=EAST
    A1: pos(walkable)=0  holding=dish  facing=NORTH
    Obs A0: self_pos=2 self_ori=2 self_held=3 other_pos=0 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=0 self_ori=0 self_held=2 other_pos=2 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=0.81, H=0.62)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.316:
        #1 [   W→N→I]                      0.012
        #2 [   W→S→S]                      0.005
        #3 [   W→N→S]                      0.005
        #4 [   W→I→I]                      0.005
        #5 [   W→I→S]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→E]                      0.011
        #2 [   S→I→N]                      0.011
        #3 [   S→I→S]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: STAY [4]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 95 ---
    Env state:  A0@2 held=soup | A1@0 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾 [38;5;208m▲ [0m  [93m▶ [0m🄾 
      🅿       🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=2  holding=soup  facing=EAST
    A1: pos(walkable)=0  holding=dish  facing=NORTH
    Obs A0: self_pos=2 self_ori=2 self_held=3 other_pos=0 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=0 self_ori=0 self_held=2 other_pos=2 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=0.81, H=0.62)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.316:
        #1 [   W→N→I]                      0.012
        #2 [   W→I→I]                      0.005
        #3 [   W→N→N]                      0.005
        #4 [   W→I→N]                      0.005
        #5 [   W→S→I]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→E]                      0.011
        #2 [   S→I→N]                      0.011
        #3 [   S→I→S]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: SOUTH [1]
    Action A1: WEST [3]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 96 ---
    Env state:  A0@5 held=soup | A1@0 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾 [38;5;208m◀ [0m    🄾 
      🅿     [93m▼ [0m🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=5  holding=soup  facing=SOUTH
    A1: pos(walkable)=0  holding=dish  facing=WEST
    Obs A0: self_pos=5 self_ori=1 self_held=3 other_pos=0 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=0 self_ori=3 self_held=2 other_pos=5 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=0.81, H=0.62)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 3 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.366:
        #1 [   W→N→I]                      0.011
        #2 [   W→I→I]                      0.005
        #3 [   W→N→S]                      0.005
        #4 [   W→I→S]                      0.005
        #5 [   W→I→N]                      0.005
    Policy beliefs A1:
      entropy 5.354:
        #1 [   S→I→N]                      0.011
        #2 [   S→I→S]                      0.011
        #3 [   S→I→E]                      0.011
        #4 [   S→I→W]                      0.011
        #5 [   S→I→S]                      0.011
    Action A0: STAY [4]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 97 ---
    Env state:  A0@5 held=soup | A1@1 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾   [38;5;208m▶ [0m  🄾 
      🅿     [93m▼ [0m🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=5  holding=soup  facing=SOUTH
    A1: pos(walkable)=1  holding=dish  facing=EAST
    Obs A0: self_pos=5 self_ori=1 self_held=3 other_pos=1 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=1 self_ori=2 self_held=2 other_pos=5 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=0.81, H=0.62)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 4 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.366:
        #1 [   W→N→I]                      0.011
        #2 [   W→S→E]                      0.005
        #3 [   W→I→S]                      0.005
        #4 [   W→N→S]                      0.005
        #5 [   W→S→N]                      0.005
    Policy beliefs A1:
      entropy 5.374:
        #1 [   N→E→I]                      0.007
        #2 [   W→S→I]                      0.007
        #3 [   S→W→W]                      0.005
        #4 [   W→S→N]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: WEST [3]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 98 ---
    Env state:  A0@4 held=soup | A1@1 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾   [38;5;208m▲ [0m  🄾 
      🅿   [93m◀ [0m  🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=4  holding=soup  facing=WEST
    A1: pos(walkable)=1  holding=dish  facing=NORTH
    Obs A0: self_pos=4 self_ori=3 self_held=3 other_pos=1 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=1 self_ori=0 self_held=2 other_pos=4 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 1 (p=0.81, H=0.62)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.256:
        #1 [   N→I→N]                      0.021
        #2 [   N→I→S]                      0.021
        #3 [   N→I→S]                      0.021
        #4 [   N→I→E]                      0.021
        #5 [   N→I→W]                      0.021
    Policy beliefs A1:
      entropy 5.354:
        #1 [   E→I→S]                      0.011
        #2 [   E→I→W]                      0.011
        #3 [   E→I→E]                      0.011
        #4 [   E→I→S]                      0.011
        #5 [   E→I→N]                      0.011
    Action A0: NORTH [0]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 99 ---
    Env state:  A0@1 held=soup | A1@2 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾   [93m▲ [0m[38;5;208m▶ [0m🄾 
      🅿       🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=1  holding=soup  facing=NORTH
    A1: pos(walkable)=2  holding=dish  facing=EAST
    Obs A0: self_pos=1 self_ori=0 self_held=3 other_pos=2 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=2 other_pos=1 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=0.81, H=0.62)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=1.00, H=-0.00)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 4.744:
        #1 [   I→E→I]                      0.018
        #2 [   I→W→I]                      0.018
        #3 [   I→N→S]                      0.018
        #4 [   I→S→E]                      0.018
        #5 [   I→N→N]                      0.018
    Policy beliefs A1:
      entropy 5.198:
        #1 [   I→E→N]                      0.012
        #2 [   I→S→S]                      0.012
        #3 [   I→S→W]                      0.012
        #4 [   I→S→S]                      0.012
        #5 [   I→S→I]                      0.012
    Action A0: INTERACT [5]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 100 ---
    Env state:  A0@1 held=soup | A1@2 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾   [93m▲ [0m[38;5;208m▶ [0m🄾 
      🅿       🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=1  holding=soup  facing=NORTH
    A1: pos(walkable)=2  holding=dish  facing=EAST
    Obs A0: self_pos=1 self_ori=0 self_held=3 other_pos=2 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=2 other_pos=1 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=0.82, H=0.60)
      pot_state 0 (p=0.99, H=0.05)
      ck_put1 0 (p=1.00, H=-0.00)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=0.90, H=0.33)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.369:
        #1 [   I→E→I]                      0.008
        #2 [   I→W→I]                      0.008
        #3 [   I→S→S]                      0.006
        #4 [   I→W→E]                      0.006
        #5 [   I→N→W]                      0.006
    Policy beliefs A1:
      entropy 5.310:
        #1 [   W→N→I]                      0.005
        #2 [   N→N→N]                      0.005
        #3 [   W→W→W]                      0.005
        #4 [   W→N→S]                      0.005
        #5 [   W→N→W]                      0.005
    Action A0: NORTH [0]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 101 ---
    Env state:  A0@1 held=soup | A1@2 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾   [93m▲ [0m[38;5;208m▲ [0m🄾 
      🅿       🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=1  holding=soup  facing=NORTH
    A1: pos(walkable)=2  holding=dish  facing=NORTH
    Obs A0: self_pos=1 self_ori=0 self_held=3 other_pos=2 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=2 self_ori=0 self_held=2 other_pos=1 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=0.82, H=0.60)
      pot_state 0 (p=1.00, H=0.00)
      ck_put1 0 (p=0.99, H=0.05)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=0.90, H=0.33)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   I→W→I]                      0.007
        #2 [   I→E→I]                      0.007
        #3 [   I→S→S]                      0.005
        #4 [   I→N→N]                      0.005
        #5 [   I→N→E]                      0.005
    Policy beliefs A1:
      entropy 5.369:
        #1 [   W→N→I]                      0.005
        #2 [   N→S→W]                      0.005
        #3 [   I→N→W]                      0.005
        #4 [   W→W→I]                      0.005
        #5 [   W→S→N]                      0.005
    Action A0: STAY [4]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 102 ---
    Env state:  A0@1 held=soup | A1@2 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾   [93m▲ [0m[38;5;208m▲ [0m🄾 
      🅿       🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=1  holding=soup  facing=NORTH
    A1: pos(walkable)=2  holding=dish  facing=NORTH
    Obs A0: self_pos=1 self_ori=0 self_held=3 other_pos=2 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=2 self_ori=0 self_held=2 other_pos=1 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 0 (p=0.82, H=0.60)
      pot_state 0 (p=1.00, H=0.00)
      ck_put1 0 (p=0.99, H=0.05)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=0.90, H=0.33)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.373:
        #1 [   I→W→I]                      0.007
        #2 [   I→E→I]                      0.007
        #3 [   I→S→S]                      0.005
        #4 [   I→S→E]                      0.005
        #5 [   I→E→E]                      0.005
    Policy beliefs A1:
      entropy 5.369:
        #1 [   W→N→I]                      0.005
        #2 [   E→W→S]                      0.005
        #3 [   W→N→W]                      0.005
        #4 [   W→N→E]                      0.005
        #5 [   W→N→S]                      0.005
    Action A0: WEST [3]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 103 ---
    Env state:  A0@0 held=soup | A1@2 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾 [93m◀ [0m  [38;5;208m▲ [0m🄾 
      🅿       🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=soup  facing=WEST
    A1: pos(walkable)=2  holding=dish  facing=NORTH
    Obs A0: self_pos=0 self_ori=3 self_held=3 other_pos=2 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=2 self_ori=0 self_held=2 other_pos=0 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=0.82, H=0.60)
      pot_state 0 (p=1.00, H=0.00)
      ck_put1 0 (p=0.99, H=0.05)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=0.90, H=0.33)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.370:
        #1 [   E→N→I]                      0.005
        #2 [   I→N→S]                      0.005
        #3 [   I→E→I]                      0.005
        #4 [   I→N→N]                      0.005
        #5 [   I→N→E]                      0.005
    Policy beliefs A1:
      entropy 5.369:
        #1 [   W→N→I]                      0.005
        #2 [   I→I→I]                      0.005
        #3 [   E→N→S]                      0.005
        #4 [   E→E→E]                      0.005
        #5 [   E→E→N]                      0.005
    Action A0: EAST [2]
    Action A1: NORTH [0]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 104 ---
    Env state:  A0@1 held=soup | A1@2 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾   [93m▶ [0m[38;5;208m▲ [0m🄾 
      🅿       🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=1  holding=soup  facing=EAST
    A1: pos(walkable)=2  holding=dish  facing=NORTH
    Obs A0: self_pos=1 self_ori=2 self_held=3 other_pos=2 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=2 self_ori=0 self_held=2 other_pos=1 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 1 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 0 (p=0.82, H=0.60)
      pot_state 0 (p=1.00, H=0.00)
      ck_put1 0 (p=0.99, H=0.05)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=0.90, H=0.33)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.375:
        #1 [   N→I→N]                      0.005
        #2 [   N→I→W]                      0.005
        #3 [   N→I→S]                      0.005
        #4 [   N→I→S]                      0.005
        #5 [   N→I→E]                      0.005
    Policy beliefs A1:
      entropy 5.369:
        #1 [   W→N→I]                      0.005
        #2 [   I→I→I]                      0.005
        #3 [   W→W→N]                      0.005
        #4 [   W→E→W]                      0.005
        #5 [   W→E→N]                      0.005
    Action A0: WEST [3]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 105 ---
    Env state:  A0@0 held=soup | A1@2 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾 [93m◀ [0m  [38;5;208m▶ [0m🄾 
      🅿       🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=soup  facing=WEST
    A1: pos(walkable)=2  holding=dish  facing=EAST
    Obs A0: self_pos=0 self_ori=3 self_held=3 other_pos=2 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=2 self_ori=2 self_held=2 other_pos=0 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 0 (p=0.82, H=0.60)
      pot_state 0 (p=1.00, H=0.00)
      ck_put1 0 (p=0.99, H=0.05)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 2 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=0.90, H=0.33)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.370:
        #1 [   E→N→I]                      0.005
        #2 [   I→N→E]                      0.005
        #3 [   I→E→I]                      0.005
        #4 [   I→N→N]                      0.005
        #5 [   I→N→S]                      0.005
    Policy beliefs A1:
      entropy 5.310:
        #1 [   W→N→I]                      0.005
        #2 [   N→N→N]                      0.005
        #3 [   W→S→E]                      0.005
        #4 [   W→E→N]                      0.005
        #5 [   W→E→W]                      0.005
    Action A0: INTERACT [5]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 106 ---
    Env state:  A0@0 held=soup | A1@5 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾 [93m◀ [0m    🄾 
      🅿     [38;5;208m▼ [0m🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=soup  facing=WEST
    A1: pos(walkable)=5  holding=dish  facing=SOUTH
    Obs A0: self_pos=0 self_ori=3 self_held=3 other_pos=5 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=5 self_ori=1 self_held=2 other_pos=0 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 3 (p=1.00, H=-0.00)
      self_held 1 (p=0.82, H=0.60)
      pot_state 0 (p=1.00, H=0.00)
      ck_put1 0 (p=0.99, H=0.05)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=0.90, H=0.33)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.321:
        #1 [   E→N→I]                      0.012
        #2 [   E→I→I]                      0.005
        #3 [   S→S→S]                      0.005
        #4 [   S→E→S]                      0.005
        #5 [   S→E→E]                      0.005
    Policy beliefs A1:
      entropy 5.375:
        #1 [   W→N→I]                      0.005
        #2 [   N→N→N]                      0.005
        #3 [   E→N→W]                      0.005
        #4 [   S→W→S]                      0.005
        #5 [   S→W→N]                      0.005
    Action A0: NORTH [0]
    Action A1: SOUTH [1]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 107 ---
    Env state:  A0@0 held=soup | A1@5 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾 [93m▲ [0m    🄾 
      🅿     [38;5;208m▼ [0m🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=soup  facing=NORTH
    A1: pos(walkable)=5  holding=dish  facing=SOUTH
    Obs A0: self_pos=0 self_ori=0 self_held=3 other_pos=5 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=5 self_ori=1 self_held=2 other_pos=0 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=0.82, H=0.60)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=0.99, H=0.05)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 1 (p=1.00, H=-0.00)
      self_held 1 (p=0.90, H=0.33)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.366:
        #1 [   E→N→I]                      0.011
        #2 [   I→I→I]                      0.005
        #3 [   E→N→S]                      0.005
        #4 [   E→S→E]                      0.005
        #5 [   E→S→S]                      0.005
    Policy beliefs A1:
      entropy 5.375:
        #1 [   W→N→I]                      0.005
        #2 [   N→N→N]                      0.005
        #3 [   E→N→W]                      0.005
        #4 [   S→W→S]                      0.005
        #5 [   S→W→N]                      0.005
    Action A0: NORTH [0]
    Action A1: EAST [2]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 108 ---
    Env state:  A0@0 held=soup | A1@5 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾 [93m▲ [0m    🄾 
      🅿     [38;5;208m▶ [0m🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=soup  facing=NORTH
    A1: pos(walkable)=5  holding=dish  facing=EAST
    Obs A0: self_pos=0 self_ori=0 self_held=3 other_pos=5 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=5 self_ori=2 self_held=2 other_pos=0 other_held=3 pot=0 delivered=0
    Beliefs A0:
      self_pos 0 (p=1.00, H=-0.00)
      self_orientation 0 (p=1.00, H=-0.00)
      self_held 1 (p=0.82, H=0.60)
      pot_state 0 (p=1.00, H=-0.00)
      ck_put1 0 (p=0.99, H=0.05)
      ck_put2 0 (p=1.00, H=-0.00)
      ck_put3 0 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 5 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 1 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 1 (p=1.00, H=-0.00)
      ctr_14 0 (p=1.00, H=-0.00)
      ctr_17 1 (p=1.00, H=-0.00)
    Beliefs A1:
      self_pos 5 (p=1.00, H=-0.00)
      self_orientation 2 (p=1.00, H=-0.00)
      self_held 1 (p=0.90, H=0.33)
      pot_state 3 (p=1.00, H=0.00)
      ck_put1 1 (p=1.00, H=-0.00)
      ck_put2 1 (p=1.00, H=-0.00)
      ck_put3 1 (p=1.00, H=-0.00)
      ck_plated 0 (p=1.00, H=-0.00)
      ck_delivered 0 (p=1.00, H=-0.00)
      other_pos 0 (p=1.00, H=-0.00)
      other_orientation 0 (p=1.00, H=-0.00)
      other_held 0 (p=1.00, H=-0.00)
      ctr_1 0 (p=1.00, H=-0.00)
      ctr_3 1 (p=1.00, H=-0.00)
      ctr_10 0 (p=1.00, H=-0.00)
      ctr_14 1 (p=1.00, H=-0.00)
      ctr_17 0 (p=1.00, H=-0.00)
    Policy beliefs A0:
      entropy 5.366:
        #1 [   E→N→I]                      0.011
        #2 [   I→I→I]                      0.005
        #3 [   E→N→S]                      0.005
        #4 [   E→S→E]                      0.005
        #5 [   E→S→S]                      0.005
    Policy beliefs A1:
      entropy 5.375:
        #1 [   W→N→I]                      0.005
        #2 [   N→N→N]                      0.005
        #3 [   E→N→W]                      0.005
        #4 [   S→W→S]                      0.005
        #5 [   S→W→N]                      0.005
    Action A0: WEST [3]
    Action A1: INTERACT [5]
    Reward A0: 0.0  (cumulative: 0.0)
    Reward A1: 0.0  (cumulative: 0.0)

  --- Step 109 ---
    Env state:  A0@0 held=soup | A1@5 held=dish | pot=empty
    Map (before action):
      🅲 🅾 ⓪ 🅾 🅲 
      🄾 [93m◀ [0m    🄾 
      🅿     [38;5;208m▶ [0m🅾 
      🅲 🄳 🅲 🅂 🅲 
    A0: pos(walkable)=0  holding=soup  facing=WEST
    A1: pos(walkable)=5  holding=dish  facing=EAST
    Obs A0: self_pos=0 self_ori=3 self_held=3 other_pos=5 other_held=2 pot=0 delivered=0
    Obs A1: self_pos=5 self_ori=2 self_held=2 other_pos=0 other_held=3 pot=0 delivered=0
