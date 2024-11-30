import copy
from typing import List, Union, overload
from collections import deque

from Chan.Bi.Bi import CBi
from Chan.Bi.BiList import CBiList
from Chan.ChanConfig import CChanConfig
from Chan.Common.CEnum import KLINE_DIR
from Chan.Common.ChanException import CChanException, ErrCode

from Chan.KLine.KLine import CKLine
from Chan.KLine.KLine_Unit import CKLine_Unit


class CKLine_List:
    DEBUG_SEG = False
    def __init__(self, kl_type, conf: CChanConfig):
        self.kl_type = kl_type
        self.config = conf
        self.lst: List[CKLine] = []  # K线列表，可递归  元素KLine类型
        
        self.new_bi_start:bool = False
        self.num_bi:int = 0
        
        self.bi_list = CBiList(bi_conf=conf.bi_conf, callback=None)
        
    def __deepcopy__(self, memo):
        new_obj = CKLine_List(self.kl_type, self.config)
        memo[id(self)] = new_obj
        for klc in self.lst:
            klus_new = []
            for klu in klc.lst:
                new_klu = copy.deepcopy(klu, memo)
                memo[id(klu)] = new_klu
                if klu.pre is not None:
                    new_klu.set_pre_klu(memo[id(klu.pre)])
                klus_new.append(new_klu)

            new_klc = CKLine(klus_new[0], idx=klc.idx, _dir=klc.dir)
            new_klc.set_fx(klc.fx)
            new_klc.kl_type = klc.kl_type
            for idx, klu in enumerate(klus_new):
                klu.set_klc(new_klc)
                if idx != 0:
                    new_klc.add(klu)
            memo[id(klc)] = new_klc
            if new_obj.lst:
                new_obj.lst[-1].set_next(new_klc)
                new_klc.set_pre(new_obj.lst[-1])
            new_obj.lst.append(new_klc)
        new_obj.bi_list = copy.deepcopy(self.bi_list, memo)
        return new_obj

    @overload
    def __getitem__(self, index: int) -> CKLine: ...

    @overload
    def __getitem__(self, index: slice) -> List[CKLine]: ...

    def __getitem__(self, index: Union[slice, int]) -> Union[List[CKLine], CKLine]:
        return self.lst[index]

    def __len__(self):
        return len(self.lst)

    def try_add_virtual_bi(self):
        self.bi_list.try_add_virtual_bi(self.lst[-1])
        
    def add_single_klu(self, klu: CKLine_Unit):
        if len(self.lst) == 0:
            self.lst.append(CKLine(klu, idx=0))
        else:
            _dir = self.lst[-1].try_add(klu)
            if _dir != KLINE_DIR.COMBINE:  # 不需要合并K线
                self.lst.append(CKLine(klu, idx=len(self.lst), _dir=_dir))
                if len(self.lst) >= 3:
                    self.lst[-2].update_fx(self.lst[-3], self.lst[-1])
                self.bi_list.update_bi(self.lst[-2], self.lst[-1], False)
            else:
                self.bi_list.try_add_virtual_bi(self.lst[-1], need_del_end=True)
                    
        # now klu(klc/combined kline) are added, bi list is also updated, now check if new bi is formed
        if self.num_bi != len(self.bi_list):
            self.new_bi_start = True
            self.num_bi = len(self.bi_list)
        else:
            self.new_bi_start = False
            
    def klu_iter(self, klc_begin_idx=0):
        for klc in self.lst[klc_begin_idx:]:
            yield from klc.lst








