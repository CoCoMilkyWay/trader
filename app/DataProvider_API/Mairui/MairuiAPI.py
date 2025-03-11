import requests
from typing import List, Dict, Tuple, Optional

from ..License import MAIRUI_LICENSE


class MairuiAPI:
    """
    A class to encapsulate Mairui APIs related to stock trading and data.
    """

    def __init__(self):
        self.LICENSE = MAIRUI_LICENSE  # Replace with your actual license key
        self.API = API = {}

        # 沪深两市股票
        API['hslt_list'] = {
            "link": "http://api.mairui.club/hslt/list/{LICENSE}",
            "description": "获取沪深两市所有股票的代码、名称、所属交易所信息。",
            "output": {
                "dm": ("string", "股票代码"),
                "mc": ("string", "股票名称"),
                "jys": ("string", "交易所"),
            }
        }

        # 沪深两市新股日历
        API['hslt_new'] = {
            "link": "http://api.mairui.club/hslt/new/{LICENSE}",
            "description": "获取沪深两市的新股日历。",
            "output": {
                "zqdm": ("string", "股票代码"),
                "zqjc": ("string", "股票简称"),
                "sgdm": ("string", "申购代码"),
                "fxsl": ("number", "发行总数（股）"),
                "swfxsl": ("number", "网上发行（股）"),
                "sgsx": ("number", "申购上限（股）"),
                "dgsz": ("number", "顶格申购需配市值(元)"),
                "sgrq": ("string", "申购日期"),
                "fxjg": ("number", "发行价格（元），null为“未知”"),
                "zxj": ("number", "最新价（元），null为“未知”"),
                "srspj": ("number", "首日收盘价（元），null为“未知”"),
                "zqgbrq": ("string", "中签号公布日，null为未知"),
                "zqjkrq": ("string", "中签缴款日，null为未知"),
                "ssrq": ("string", "上市日期，null为未知"),
                "syl": ("number", "发行市盈率，null为“未知”"),
                "hysyl": ("number", "行业市盈率"),
                "wszql": ("number", "中签率（%），null为“未知”"),
                "yzbsl": ("number", "连续一字板数量，null为“未知”"),
                "zf": ("number", "涨幅（%），null为“未知”"),
                "yqhl": ("number", "每中一签获利（元），null为“未知”"),
                "zyyw": ("string", "主营业务"),
            }
        }

        # 指数、行业、概念
        API['hszg_list'] = {
            "link": "http://api.mairui.club/hszg/list/{LICENSE}",
            "description": "获取指数、行业、概念的代码。",
            "output": {
                "code": ("string", "代码"),
                "name": ("string", "名称"),
                "type1": ("number", "一级分类"),
                "type2": ("number", "二级分类"),
                "level": ("number", "层级"),
                "pcode": ("string", "父节点代码"),
                "pname": ("string", "父节点名称"),
                "isleaf": ("number", "是否为叶子节点"),
            }
        }

        # 根据指数、行业、概念找相关股票
        API['hszg_gg'] = {
            "link": "http://api.mairui.club/hszg/gg/{0}/{LICENSE}",
            "description": "根据代码找到相关的股票。",
            "output": {
                "dm": ("string", "代码"),
                "mc": ("string", "名称"),
                "jys": ("string", "交易所"),
            }
        }

        # 根据股票找相关指数、行业、概念
        API['hszg_zg'] = {
            "link": "http://api.mairui.club/hszg/zg/{0}/{LICENSE}",
            "description": "根据股票代码得到相关的指数、行业、概念。",
            "output": {
                "code": ("string", "指数、行业、概念代码"),
                "name": ("string", "指数、行业、概念名称"),
            }
        }

        # 公司简介
        API['hscp_gsjj'] = {
            "link": "http://api.mairui.club/hscp/gsjj/{0}/{LICENSE}",
            "description": "获取上市公司的简介。",
            "output": {
                "name": ("string", "公司名称"),
                "ename": ("string", "公司英文名称"),
                "market": ("string", "上市市场"),
                "idea": ("string", "概念及板块"),
                "ldate": ("string", "上市日期"),
                "sprice": ("string", "发行价格（元）"),
                "principal": ("string", "主承销商"),
                "rdate": ("string", "成立日期"),
                "rprice": ("string", "注册资本"),
                "instype": ("string", "机构类型"),
                "organ": ("string", "组织形式"),
                "secre": ("string", "董事会秘书"),
                "phone": ("string", "公司电话"),
                "sphone": ("string", "董秘电话"),
                "fax": ("string", "公司传真"),
                "sfax": ("string", "董秘传真"),
                "email": ("string", "公司电子邮箱"),
                "semail": ("string", "董秘电子邮箱"),
                "site": ("string", "公司网站"),
                "post": ("string", "邮政编码"),
                "infosite": ("string", "信息披露网址"),
                "oname": ("string", "证券简称更名历史"),
                "addr": ("string", "注册地址"),
                "oaddr": ("string", "办公地址"),
                "desc": ("string", "公司简介"),
                "bscope": ("string", "经营范围"),
                "printype": ("string", "承销方式"),
                "referrer": ("string", "上市推荐人"),
                "putype": ("string", "发行方式"),
                "pe": ("number", "发行市盈率"),
                "firgu": ("number", "首发前总股本"),
                "lastgu": ("number", "首发后总股本"),
                "realgu": ("number", "实际发行量"),
                "planm": ("number", "预计募集资金"),
                "realm": ("number", "实际募集资金合计"),
                "pubfee": ("number", "发行费用总额"),
                "collect": ("number", "募集资金净额"),
                "signfee": ("number", "承销费用"),
                "pdate": ("string", "招股公告日"),
            }
        }

        # 所属指数
        API['hscp_sszs'] = {
            "link": "http://api.mairui.club/hscp/sszs/{0}/{LICENSE}",
            "description": "获取上市公司的所属指数。",
            "output": {
                "mc": ("string", "指数名称"),
                "dm": ("string", "指数代码"),
                "ind": ("string", "进入日期"),
                "outd": ("string", "退出日期"),
            }
        }

        # 历届高管成员
        API['hscp_ljgg'] = {
            "link": "http://api.mairui.club/hscp/ljgg/{0}/{LICENSE}",
            "description": "获取上市公司的历届高管成员名单。",
            "output": {
                "name": ("string", "姓名"),
                "title": ("string", "职务"),
                "sdate": ("string", "起始日期"),
                "edate": ("string", "终止日期"),
            }
        }

        # 历届董事会成员
        API['hscp_ljds'] = {
            "link": "http://api.mairui.club/hscp/ljds/{0}/{LICENSE}",
            "description": "获取上市公司的历届董事会成员名单。",
            "output": {
                "name": ("string", "姓名"),
                "title": ("string", "职务"),
                "sdate": ("string", "起始日期"),
                "edate": ("string", "终止日期"),
            }
        }

        # 历届监事会成员
        API['hscp_ljjj'] = {
            "link": "http://api.mairui.club/hscp/ljjj/{0}/{LICENSE}",
            "description": "获取上市公司的历届监事会成员名单。",
            "output": {
                "name": ("string", "姓名"),
                "title": ("string", "职务"),
                "sdate": ("string", "起始日期"),
                "edate": ("string", "终止日期"),
            }
        }

        # 近年分红
        API['hscp_jnfh'] = {
            "link": "http://api.mairui.club/hscp/jnfh/{0}/{LICENSE}",
            "description": "获取上市公司的近年来的分红实施结果。",
            "output": {
                "sdate": ("string", "公告日期"),
                "give": ("string", "每10股送股"),
                "change": ("string", "每10股转增"),
                "send": ("string", "每10股派息（税前）"),
                "line": ("string", "进度"),
                "cdate": ("string", "除权除息日"),
                "edate": ("string", "股权登记日"),
                "hdate": ("string", "红股上市日"),
            }
        }

        # 近年增发
        API['hscp_jnzf'] = {
            "link": "http://api.mairui.club/hscp/jnzf/{0}/{LICENSE}",
            "description": "获取上市公司的近年来的增发情况。",
            "output": {
                "sdate": ("string", "公告日期"),
                "type": ("string", "发行方式"),
                "price": ("string", "发行价格"),
                "tprice": ("string", "实际募集资金总额"),
                "fprice": ("string", "发行费用总额"),
                "amount": ("string", "实际发行数量"),
            }
        }

        # 解禁限售
        API['hscp_jjxs'] = {
            "link": "http://api.mairui.club/hscp/jjxs/{0}/{LICENSE}",
            "description": "获取上市公司的解禁限售情况。",
            "output": {
                "rdate": ("string", "解禁日期"),
                "ramount": ("number", "解禁数量"),
                "rprice": ("number", "解禁股流通市值"),
                "batch": ("number", "上市批次"),
                "pdate": ("string", "公告日期"),
            }
        }

        # 近一年各季度利润
        API['hscp_jdlr'] = {
            "link": "http://api.mairui.club/hscp/jdlr/{0}/{LICENSE}",
            "description": "获取上市公司近一年各个季度的利润。",
            "output": {
                "date": ("string", "截止日期"),
                "income": ("string", "营业收入"),
                "expend": ("string", "营业支出"),
                "profit": ("string", "营业利润"),
                "totalp": ("string", "利润总额"),
                "reprofit": ("string", "净利润"),
                "basege": ("string", "基本每股收益"),
                "ettege": ("string", "稀释每股收益"),
                "otherp": ("string", "其他综合收益"),
                "totalcp": ("string", "综合收益总额"),
            }
        }

        # 近一年各季度现金流
        API['hscp_jdxj'] = {
            "link": "http://api.mairui.club/hscp/jdxj/{0}/{LICENSE}",
            "description": "获取上市公司近一年各个季度的现金流。",
            "output": {
                "date": ("string", "截止日期"),
                "jyin": ("string", "经营活动现金流入"),
                "jyout": ("string", "经营活动现金流出"),
                "jyfinal": ("string", "经营活动产生的现金流量净额"),
                "tzin": ("string", "投资活动现金流入"),
                "tzout": ("string", "投资活动现金流出"),
                "tzfinal": ("string", "投资活动产生的现金流量净额"),
                "czin": ("string", "筹资活动现金流入"),
                "czout": ("string", "筹资活动现金流出"),
                "czfinal": ("string", "筹资活动产生的现金流量净额"),
                "hl": ("string", "汇率变动影响"),
                "cashinc": ("string", "现金及现金等价物净增加额"),
                "cashs": ("string", "期初现金及现金等价物余额"),
                "cashe": ("string", "期末现金及现金等价物余额"),
            }
        }

        # 近年业绩预告
        API['hscp_yjyg'] = {
            "link": "http://api.mairui.club/hscp/yjyg/{0}/{LICENSE}",
            "description": "获取上市公司近年来的业绩预告。",
            "output": {
                "pdate": ("string", "公告日期"),
                "rdate": ("string", "报告期"),
                "type": ("string", "类型"),
                "abs": ("string", "业绩预告摘要"),
                "old": ("string", "上年同期每股收益"),
            }
        }

        # 财务指标
        API['hscp_cwzb'] = {
            "link": "http://api.mairui.club/hscp/cwzb/{0}/{LICENSE}",
            "description": "获取上市公司近四个季度的主要财务指标。",
            "output": {
                "date": ("string", "报告日期"),
                "tbmg": ("string", "摊薄每股收益"),
                "jqmg": ("string", "加权每股收益"),
                "mgsy": ("string", "每股收益_调整后"),
                "kfmg": ("string", "扣除非经常性损益后的每股收益"),
                "mgjz": ("string", "每股净资产_调整前"),
                "mgjzad": ("string", "每股净资产_调整后"),
                "mgjy": ("string", "每股经营性现金流"),
                "mggjj": ("string", "每股资本公积金"),
                "mgwly": ("string", "每股未分配利润"),
                "zclr": ("string", "总资产利润率"),
                "zylr": ("string", "主营业务利润率"),
                "zzlr": ("string", "总资产净利润率"),
                "cblr": ("string", "成本费用利润率"),
                "yylr": ("string", "营业利润率"),
                "zycb": ("string", "主营业务成本率"),
                "xsjl": ("string", "销售净利率"),
                "gbbc": ("string", "股本报酬率"),
                "jzbc": ("string", "净资产报酬率"),
                "zcbc": ("string", "资产报酬率"),
                "xsml": ("string", "销售毛利率"),
                "xxbz": ("string", "三项费用比重"),
                "fzy": ("string", "非主营比重"),
                "zybz": ("string", "主营利润比重"),
                "gxff": ("string", "股息发放率"),
                "tzsy": ("string", "投资收益率"),
                "zyyw": ("string", "主营业务利润"),
                "jzsy": ("string", "净资产收益率"),
                "jqjz": ("string", "加权净资产收益率"),
                "kflr": ("string", "扣除非经常性损益后的净利润"),
                "zysr": ("string", "主营业务收入增长率"),
                "jlzz": ("string", "净利润增长率"),
                "jzzz": ("string", "净资产增长率"),
                "zzzz": ("string", "总资产增长率"),
                "yszz": ("string", "应收账款周转率"),
                "yszzt": ("string", "应收账款周转天数"),
                "chzz": ("string", "存货周转天数"),
                "chzzl": ("string", "存货周转率"),
                "gzzz": ("string", "固定资产周转率"),
                "zzzzl": ("string", "总资产周转率"),
                "zzzzt": ("string", "总资产周转天数"),
                "ldzz": ("string", "流动资产周转率"),
                "ldzzt": ("string", "流动资产周转天数"),
                "gdzz": ("string", "股东权益周转率"),
                "ldbl": ("string", "流动比率"),
                "sdbl": ("string", "速动比率"),
                "xjbl": ("string", "现金比率"),
                "lxzf": ("string", "利息支付倍数"),
                "zjbl": ("string", "长期债务与营运资金比率"),
                "gdqy": ("string", "股东权益比率"),
                "cqfz": ("string", "长期负债比率"),
                "gdgd": ("string", "股东权益与固定资产比率"),
                "fzqy": ("string", "负债与所有者权益比率"),
                "zczjbl": ("string", "长期资产与长期资金比率"),
                "zblv": ("string", "资本化比率"),
                "gdzcjz": ("string", "固定资产净值率"),
                "zbgdh": ("string", "资本固定化比率"),
                "cqbl": ("string", "产权比率"),
                "qxjzb": ("string", "清算价值比率"),
                "gdzcbz": ("string", "固定资产比重"),
                "zcfzl": ("string", "资产负债率"),
                "zzc": ("string", "总资产"),
                "jyxj": ("string", "经营现金净流量对销售收入比率"),
                "zcjyxj": ("string", "资产的经营现金流量回报率"),
                "jylrb": ("string", "经营现金净流量与净利润的比率"),
                "jyfzl": ("string", "经营现金净流量对负债比率"),
                "xjlbl": ("string", "现金流量比率"),
                "dqgptz": ("string", "短期股票投资"),
                "dqzctz": ("string", "短期债券投资"),
                "dqjytz": ("string", "短期其它经营性投资"),
                "qcgptz": ("string", "长期股票投资"),
                "cqzqtz": ("string", "长期债券投资"),
                "cqjyxtz": ("string", "长期其它经营性投资"),
                "yszk1": ("string", "1年以内应收帐款"),
                "yszk12": ("string", "1-2年以内应收帐款"),
                "yszk23": ("string", "2-3年以内应收帐款"),
                "yszk3": ("string", "3年以内应收帐款"),
                "yfhk1": ("string", "1年以内预付货款"),
                "yfhk12": ("string", "1-2年以内预付货款"),
                "yfhk23": ("string", "2-3年以内预付货款"),
                "yfhk3": ("string", "3年以内预付货款"),
                "ysk1": ("string", "1年以内其它应收款"),
                "ysk12": ("string", "1-2年以内其它应收款"),
                "ysk23": ("string", "2-3年以内其它应收款"),
                "ysk3": ("string", "3年以内其它应收款"),
            }
        }

        # 十大股东
        API['hscp_sdgd'] = {
            "link": "http://api.mairui.club/hscp/sdgd/{0}/{LICENSE}",
            "description": "获取上市公司的十大股东数据。",
            "output": {
                "jzrq": ("string", "截止日期"),
                "ggrq": ("string", "公告日期"),
                "gdsm": ("string", "股东说明"),
                "gdzs": ("number", "股东总数"),
                "pjcg": ("number", "平均持股"),
            }
        }

        # 十大流通股东
        API['hscp_ltgd'] = {
            "link": "http://api.mairui.club/hscp/ltgd/{0}/{LICENSE}",
            "description": "获取上市公司的十大流通股东数据。",
            "output": {
                "jzrq": ("string", "截止日期"),
                "ggrq": ("string", "公告日期"),
                "gdsm": ("string", "股东说明"),
                "gdzs": ("number", "股东总数"),
                "pjcg": ("number", "平均持股"),
            }
        }

        # 股东变化趋势
        API['hscp_gdbh'] = {
            "link": "http://api.mairui.club/hscp/gdbh/{0}/{LICENSE}",
            "description": "获取上市公司的股东变化趋势数据。",
            "output": {
                "jzrq": ("string", "截止日期"),
                "gdhs": ("number", "股东户数"),
                "bh": ("string", "比上期变化情况"),
            }
        }

        # 基金持股
        API['hscp_jjcg'] = {
            "link": "http://api.mairui.club/hscp/jjcg/{0}/{LICENSE}",
            "description": "获取该股票最近500家左右的基金持股情况。",
            "output": {
                "jzrq": ("string", "截止日期"),
                "jjmc": ("string", "基金名称"),
                "jjdm": ("string", "基金代码"),
                "ccsl": ("number", "持仓数量"),
                "ltbl": ("number", "占流通股比例"),
                "cgsz": ("number", "持股市值"),
                "jzbl": ("number", "占净值比例"),
            }
        }

        # 主力资金走势
        API['hsmy_zlzj'] = {
            "link": "http://api.mairui.club/hsmy/zlzj/{0}/{LICENSE}",
            "description": "得到每分钟主力资金走势图。",
            "output": {
                "t": ("string", "时间"),
                "zdf": ("number", "涨跌幅"),
                "lrzj": ("number", "主力流入"),
                "lrl": ("number", "主力流入率"),
                "lczj": ("number", "主力流出"),
                "jlr": ("number", "主力净流入"),
                "jlrl": ("number", "主力净流入率"),
                "shlrl": ("number", "散户流入率"),
            }
        }

        # 资金流入趋势
        API['hsmy_zjlr'] = {
            "link": "http://api.mairui.club/hsmy/zjlr/{0}/{LICENSE}",
            "description": "得到近十年的资金流入趋势。",
            "output": {
                "t": ("string", "时间"),
                "zdf": ("number", "涨跌幅"),
                "hsl": ("number", "换手率"),
                "jlr": ("number", "净流入"),
                "jlrl": ("number", "净流入率"),
                "zljlr": ("number", "主力净流入"),
                "zljlrl": ("number", "主力净流入率"),
                "hyjlr": ("number", "行业净流入"),
                "hyjlrl": ("number", "行业净流入率"),
            }
        }

        # 最近10天资金流入趋势
        API['hsmy_zhlrt'] = {
            "link": "http://api.mairui.club/hsmy/zhlrt/{0}/{LICENSE}",
            "description": "得到最近10天资金流入趋势。",
            "output": {
                "t": ("string", "时间"),
                "zdf": ("number", "涨跌幅"),
                "hsl": ("number", "换手率"),
                "jlr": ("number", "净流入"),
                "jlrl": ("number", "净流入率"),
                "zljlr": ("number", "主力净流入"),
                "zljlrl": ("number", "主力净流入率"),
                "hyjlr": ("number", "行业净流入"),
                "hyjlrl": ("number", "行业净流入率"),
            }
        }

        # 阶段主力动向
        API['hsmy_jddx'] = {
            "link": "http://api.mairui.club/hsmy/jddx/{0}/{LICENSE}",
            "description": "得到近10天阶段主力动向。",
            "output": {
                "t": ("string", "时间"),
                "jlr3": ("number", "近3日主力净流入"),
                "jlr5": ("number", "近5日主力净流入"),
                "jlr10": ("number", "近10日主力净流入"),
                "jlrl3": ("number", "近3日主力净流入率"),
                "jlrl5": ("number", "近5日主力净流入率"),
                "jlrl10": ("number", "近10日主力净流入率"),
            }
        }

        # 最近10天阶段主力动向
        API['hsmy_jddxt'] = {
            "link": "http://api.mairui.club/hsmy/jddxt/{0}/{LICENSE}",
            "description": "得到最近10天阶段主力动向。",
            "output": {
                "t": ("string", "时间"),
                "jlr3": ("number", "近3日主力净流入"),
                "jlr5": ("number", "近5日主力净流入"),
                "jlr10": ("number", "近10日主力净流入"),
                "jlrl3": ("number", "近3日主力净流入率"),
                "jlrl5": ("number", "近5日主力净流入率"),
                "jlrl10": ("number", "近10日主力净流入率"),
            }
        }

        # 历史成交分布
        API['hsmy_lscj'] = {
            "link": "http://api.mairui.club/hsmy/lscj/{0}/{LICENSE}",
            "description": "得到近十年每天历史成交分布。",
            "output": {
                "t": ("string", "时间"),
                "c": ("number", "收盘价"),
                "zdf": ("number", "涨跌幅"),
                "jlrl": ("number", "净流入率"),
                "hsl": ("number", "换手率"),
                "qbjlr": ("number", "全部净流入"),
                "cddlr": ("number", "超大单流入"),
                "cddjlr": ("number", "超大单净流入"),
                "ddlr": ("number", "大单流入"),
                "ddjlr": ("number", "大单净流入"),
                "xdlr": ("number", "小单流入"),
                "xdjlr": ("number", "小单净流入"),
                "sdlr": ("number", "散单流入"),
                "sdjlr": ("number", "散单净流入"),
            }
        }

        # 最近10天成交分布
        API['hsmy_lscjt'] = {
            "link": "http://api.mairui.club/hsmy/lscjt/{0}/{LICENSE}",
            "description": "得到最近10天成交分布。",
            "output": {
                "t": ("string", "时间"),
                "c": ("number", "收盘价"),
                "zdf": ("number", "涨跌幅"),
                "jlrl": ("number", "净流入率"),
                "hsl": ("number", "换手率"),
                "qbjlr": ("number", "全部净流入"),
                "cddlr": ("number", "超大单流入"),
                "cddjlr": ("number", "超大单净流入"),
                "ddlr": ("number", "大单流入"),
                "ddjlr": ("number", "大单净流入"),
                "xdlr": ("number", "小单流入"),
                "xdjlr": ("number", "小单净流入"),
                "sdlr": ("number", "散单流入"),
                "sdjlr": ("number", "散单净流入"),
            }
        }

        # 涨停股池
        API['hslt_ztgc'] = {
            "link": "http://api.mairui.club/hslt/ztgc/{0}/{LICENSE}",
            "description": "根据日期得到每天的涨停股票列表。",
            "output": {
                "dm": ("string", "代码"),
                "mc": ("string", "名称"),
                "p": ("number", "价格"),
                "zf": ("number", "涨幅"),
                "cje": ("number", "成交额"),
                "lt": ("number", "流通市值"),
                "zsz": ("number", "总市值"),
                "hs": ("number", "换手率"),
                "lbc": ("number", "连板数"),
                "fbt": ("string", "首次封板时间"),
                "lbt": ("string", "最后封板时间"),
                "zj": ("number", "封板资金"),
                "zbc": ("number", "炸板次数"),
                "tj": ("string", "涨停统计"),
            }
        }

        # 跌停股池
        API['hslt_dtgc'] = {
            "link": "http://api.mairui.club/hslt/dtgc/{0}/{LICENSE}",
            "description": "根据日期得到每天的跌停股票列表。",
            "output": {
                "dm": ("string", "代码"),
                "mc": ("string", "名称"),
                "p": ("number", "价格"),
                "zf": ("number", "跌幅"),
                "cje": ("number", "成交额"),
                "lt": ("number", "流通市值"),
                "zsz": ("number", "总市值"),
                "pe": ("number", "动态市盈率"),
                "hs": ("number", "换手率"),
                "lbc": ("number", "连续跌停次数"),
                "lbt": ("string", "最后封板时间"),
                "zj": ("number", "封单资金"),
                "fba": ("number", "板上成交额"),
                "zbc": ("number", "开板次数"),
            }
        }

        # 强势股池
        API['hslt_qsgc'] = {
            "link": "http://api.mairui.club/hslt/qsgc/{0}/{LICENSE}",
            "description": "根据日期得到每天的强势股票列表。",
            "output": {
                "dm": ("string", "代码"),
                "mc": ("string", "名称"),
                "p": ("number", "价格"),
                "ztp": ("number", "涨停价"),
                "zf": ("number", "涨幅"),
                "cje": ("number", "成交额"),
                "lt": ("number", "流通市值"),
                "zsz": ("number", "总市值"),
                "zs": ("number", "涨速"),
                "nh": ("number", "是否新高"),
                "lb": ("number", "量比"),
                "hs": ("number", "换手率"),
                "tj": ("string", "涨停统计"),
            }
        }

        # 次新股池
        API['hslt_cxgc'] = {
            "link": "http://api.mairui.club/hslt/cxgc/{0}/{LICENSE}",
            "description": "根据日期得到每天的次新股票列表。",
            "output": {
                "dm": ("string", "代码"),
                "mc": ("string", "名称"),
                "p": ("number", "价格"),
                "ztp": ("number", "涨停价"),
                "zf": ("number", "涨跌幅"),
                "cje": ("number", "成交额"),
                "lt": ("number", "流通市值"),
                "zsz": ("number", "总市值"),
                "nh": ("number", "是否新高"),
                "hs": ("number", "转手率"),
                "tj": ("string", "涨停统计"),
            }
        }

        # 炸板股池
        API['hslt_zbgc'] = {
            "link": "http://api.mairui.club/hslt/zbgc/{0}/{LICENSE}",
            "description": "根据日期得到每天的炸板股票列表。",
            "output": {
                "dm": ("string", "代码"),
                "mc": ("string", "名称"),
                "p": ("number", "价格"),
                "ztp": ("number", "涨停价"),
                "zf": ("number", "涨跌幅"),
                "cje": ("number", "成交额"),
                "lt": ("number", "流通市值"),
                "zsz": ("number", "总市值"),
                "zs": ("number", "涨速"),
                "hs": ("number", "转手率"),
                "tj": ("string", "涨停统计"),
                "fbt": ("string", "首次封板时间"),
                "zbc": ("number", "炸板次数"),
            }
        }

        # 融资融券标的股
        API['hsrq_list'] = {
            "link": "http://api.mairui.club/hsrq/list/{LICENSE}",
            "description": "沪深两市融资融券标的股。",
            "output": {
                "dm": ("string", "股票代码"),
                "mc": ("string", "股票名称"),
                "jys": ("string", "交易所"),
            }
        }

        # 融资融券历史走势
        API['hsrq_ls'] = {
            "link": "http://api.mairui.club/hsrq/lszs/{0}/{LICENSE}",
            "description": "根据股票代码得到近两年融资融券变化走势。",
            "output": {
                "rq": ("string", "交易日期"),
                "rzrqye": ("number", "当日融资融券余额"),
                "rzrqyecz": ("number", "当日融资融券余额差值"),
                "p": ("number", "当日收盘价"),
                "zdf1": ("number", "当日涨跌幅"),
                "zdf3": ("number", "3日涨跌幅"),
                "zdf5": ("number", "5日涨跌幅"),
                "zdf10": ("number", "10日涨跌幅"),
                "rzye": ("number", "融资当日余额"),
                "rzyezb": ("number", "融资当日余额占流通市值比"),
                "rzmre1": ("number", "融资当日买入额"),
                "rzmre3": ("number", "融资3日买入额"),
                "rzmre5": ("number", "融资5日买入额"),
                "rzmre10": ("number", "融资10日买入额"),
                "rzche1": ("number", "融资当日偿还额"),
                "rzche3": ("number", "融资3日偿还额"),
                "rzche5": ("number", "融资5日偿还额"),
                "rzche10": ("number", "融资10日偿还额"),
                "rzjm1": ("number", "融资当日净买入额"),
                "rzjm3": ("number", "融资3日净买入额"),
                "rzjm5": ("number", "融资5日净买入额"),
                "rzjm10": ("number", "融资10日净买入额"),
                "rqye": ("number", "融券当日余额"),
                "rqyl": ("number", "融券当日余量"),
                "rqmcl1": ("number", "融券当日卖出量"),
                "rqmcl3": ("number", "融券3日卖出量"),
                "rqmcl5": ("number", "融券5日卖出量"),
                "rqmcl10": ("number", "融券10日卖出量"),
                "rqchl1": ("number", "融券当日偿还量"),
                "rqchl3": ("number", "融券3日偿还量"),
                "rqchl5": ("number", "融券5日偿还量"),
                "rqchl10": ("number", "融券10日偿还量"),
                "rqjmc1": ("number", "融券当日净卖出量"),
                "rqjmc3": ("number", "融券3日净卖出量"),
                "rqjmc5": ("number", "融券5日净卖出量"),
                "rqjmc10": ("number", "融券10日净卖出量"),
            }
        }

    def query(self, name: str, inputs: List[str] = []):
        url = self.API[name]['link'].format(*inputs, LICENSE=self.LICENSE)
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            assert False, f"Error: {response.status_code} - {response.text}"
        