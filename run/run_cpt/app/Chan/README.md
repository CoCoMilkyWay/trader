```
[CChan Initialization]─────────────────────────────┐
   |                                               |
   ├── Input Validation                            |
   |    ├── lv_list (default=[K_DAY, K_60M])       |
   |    └── check_kltype_order (high to low)       |
   |                                               |
   ├── Time Handling                               |
   |    ├── Convert datetime.date to str           |
   |    └── Store begin_time/end_time              |
   |                                               |
   ├── State Initialization                        |
   |    ├── new_bi_start = False                   |
   |    ├── kl_misalign_cnt = 0                    |
   |    └── kl_inconsistent_detail = defaultdict   |
   |                                               |
   └── KLine Lists Creation [do_init()]            |
        └── Create CKLine_List for each level      |
            in kl_datas dictionary                 |
                                                   |
[Trigger Load Process]◄────────────────────────────┘
   |
   ├── Initialize if needed
   |    ├── klu_cache = [None] * len(lv_list)
   |    └── klu_last_t = [CTime(1980,1,1,0,0)] * len(lv_list)
   |
   ├── Input Processing
   |    ├── Check highest level data presence
   |    └── Convert input lists to iterators
   |
   └── Start Loading Loop
        |
        v
[Load Iterator Core]─────────────────────┐
   |                                     |
   ├── Variables Per Level               |
   |    ├── cur_lv = lv_list[lv_idx]     |
   |    └── pre_klu tracking             |
   |                                     |
   ├── KLine Unit Processing             |
   |    |                                |
   |    ├── [Cache Check]                |
   |    |    ├── Use cached KLU if exists|
   |    |    └── Clear cache after use   |
   |    |                                |
   |    ├── [New KLU Retrieval]          |
   |    |    ├── get_next_lv_klu()       |
   |    |    ├── Set KLU index           |
   |    |    └── Time monotonicity check |
   |    |                                |
   |    └── [Parent Time Check]          |
   |         ├── Compare with parent time|
   |         └── Cache if beyond parent  |
   |                                     |
   ├── KLU Integration                   |
   |    ├── Link with previous KLU       |
   |    ├── Add to KLine List            |
   |    └── Set parent relations         |
   |                                     |
   └── Multi-Level Processing            |
        |                                |
        ├── [Parent-Child Relations]     |
        |    ├── Time consistency check  |
        |    ├── Add child to parent     |
        |    └── Set parent reference    |
        |                                |
        ├── [Recursive Processing]       |
        |    ├── Process lower levels    |
        |    └── Check alignment         |
        |                                |
        └── [Validation]                 |
             ├── Alignment check         |
             └── Consistency check       |

[Validation System]
   |
   ├── Alignment Check
   |    ├── Verify sub_kl_list presence
   |    ├── Track misalignment count
   |    └── Throw if misaligned
   |
   ├── Time Consistency
   |    ├── Compare Y/M/D components
   |    ├── Track inconsistencies
   |    └── Throw if inconsistent
   |
   └── Index Management
        ├── Check existing index
        ├── Initialize if first
        └── Increment from last

[Error Handling]
   |
   ├── KL_NOT_MONOTONOUS
   ├── KL_DATA_NOT_ALIGN
   ├── KL_TIME_INCONSISTENT
   └── NO_DATA

Legend:
───  Flow direction
 |   Vertical connection
 v   Flow continues
 ├   Branch point
 └   Last branch
[ ]  Major process group
◄──  Connection/Return
```