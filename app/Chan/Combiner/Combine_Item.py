from Chan.Common.ChanException import CChanException, ErrCode


class CCombine_Item:
    def __init__(self, item):
        from Chan.Bi.Bi import CBi
        from Chan.KLine.KLine_Unit import CKLine_Unit
        if type(item) == CBi:
            self.time_begin = item.begin_klc.idx
            self.time_end = item.end_klc.idx
            self.high = item._high()
            self.low = item._low()
        elif type(item) == CKLine_Unit:
            self.time_begin = item.time
            self.time_end = item.time
            self.high = item.high
            self.low = item.low
        else:
            raise CChanException(f"{type(item)} is unsupport sub class of CCombine_Item", ErrCode.COMMON_ERROR)
