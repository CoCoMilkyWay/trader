# coding:utf-8

def download_metatable_data():
    '''
    下载metatable信息
    通常在客户端启动时自动获取，不需要手工调用
    '''
    from .. import xtdata
    cl = xtdata.get_client()

    ret = xtdata._BSON_call_common(
        cl.commonControl, 'downloadmetatabledata', {}
    )
    return ret


__META_INFO__ = {}
__META_FIELDS__ = {}
__META_TABLES__ = {}


def _init_metainfos():
    '''
    初始化metatable
    '''
    import traceback
    from .. import xtdata, xtbson

    global __META_INFO__
    global __META_FIELDS__
    global __META_TABLES__

    cl = xtdata.get_client()
    result = xtbson.BSON.decode(cl.commonControl('getmetatabledatas', xtbson.BSON.encode({})))
    all_metainfos = result['result']

    for metainfo in all_metainfos:
        if not isinstance(metainfo, dict):
            continue

        try:
            metaid = metainfo['I']
            __META_INFO__[metaid] = metainfo

            table_name = metainfo.get('modelName', metaid)
            table_name_cn = metainfo.get('tableNameCn', '')

            __META_TABLES__[table_name] = metaid
            __META_TABLES__[table_name_cn] = metaid

            metainfo_fields = metainfo.get('fields', {})
            # metainfo_fields.pop('G', None)  # G公共时间字段特殊处理，跳过
            for key, info in metainfo_fields.items():
                field_name = info['modelName']
                __META_FIELDS__[f'{table_name}.{field_name}'] = (metaid, key)
        except:
            traceback.print_exc()
            continue
    return


def get_metatable_list():
    '''
    获取metatable列表

    return:
        { table_code1: table_name1, table_code2: table_name2, ... }

        table_code: str
            数据表代码
        table_name: str
            数据表名称
    '''
    if not __META_INFO__:
        _init_metainfos()

    ret = {}
    for metaid, metainfo in __META_INFO__.items():
        model_name = metainfo.get('modelName', f'{metaid}')
        table_name_desc = metainfo.get('tableNameCn', '')
        ret[model_name] = table_name_desc

    return ret


def get_metatable_config(table):
    '''
    获取metatable列表原始配置信息
    '''
    if not __META_INFO__:
        _init_metainfos()

    if table not in __META_TABLES__:
        print(f'[ERROR] Unknown table {table}')

    metaid = __META_TABLES__[table]
    return __META_INFO__[metaid]


__META_TYPECONV__ = {
    'int': int(),
    'long': int(),
    'double': float(),
    'string': str(),
    'binary': bytes(),
}


def _meta_type(t):
    try:
        return __META_TYPECONV__[t]
    except:
        raise Exception(f'Unsupported type:{t}')


def get_metatable_info(table):
    '''
    获取metatable数据表信息

    table: str
        数据表代码 table_code 或 数据表名称 table_name
    return: dict
        {
            'code': table_code
            , 'name': table_name
            , 'desc': desc
            , 'fields': fields
        }

        table_code: str
            数据表代码
        table_name: str
            数据表名称
        desc: str
            描述
        fields: dict
            { 'code': field_code, 'name': field_name, 'type': field_type }
    '''
    info = get_metatable_config(table)

    fields = info.get('fields', {})
    ret = {
        'code': info.get('modelName', ''),
        'name': info.get('tableNameCn', ''),
        'desc': info.get('desc', ''),
        'fields': [
            {
                'code': field_info.get('modelName', ''),
                'name': field_info.get('fieldNameCn', ''),
                'type': type(_meta_type(field_info.get('type', ''))),
            } for key, field_info in fields.items()
        ],
    }
    return ret


def get_metatable_fields(table):
    '''
    获取metatable数据表字段信息

    table: str
        数据表代码 table_code 或 数据表名称 table_name
    return: pd.DataFrame
        columns = ['code', 'name', 'type']
    '''
    import pandas as pd
    info = get_metatable_config(table)

    fields = info.get('fields', {})
    ret = pd.DataFrame([{
        'code': field_info.get('modelName', ''),
        'name': field_info.get('fieldNameCn', ''),
        'type': type(_meta_type(field_info.get('type', ''))),
    } for key, field_info in fields.items()])
    return ret


def parse_request_from_fields(fields):
    '''
    根据字段解析metaid和field
    '''
    table_field = {}  # {metaid: {key}}
    key2field = {}  # {metaid: {key: field}}
    columns = []  # table.field
    if not __META_FIELDS__:
        _init_metainfos()

    for field in fields:
        if field.find('.') == -1:  # 获取整个table的数据
            metaid = __META_TABLES__[field]
            if metaid in __META_INFO__:
                metainfo = __META_INFO__[metaid]
                table = metainfo['modelName']
                meta_table_fields = metainfo.get('fields', {})
                if not meta_table_fields:
                    continue

                table_field[metaid] = {k: _meta_type(v['type']) for k, v in meta_table_fields.items()}
                key2field[metaid] = {
                    key: f'{table}.{field_info["modelName"]}' for key, field_info in meta_table_fields.items()
                }
                columns.extend(key2field[metaid].values())

        elif field in __META_FIELDS__:
            metaid, key = __META_FIELDS__[field]
            metainfo = __META_INFO__[metaid]

            if metaid not in table_field:
                table_field[metaid] = {}
            table_field[metaid][key] = _meta_type(metainfo['fields'][key]['type'])

            if metaid not in key2field:
                key2field[metaid] = {}
            key2field[metaid][key] = field

            columns.append(field)

    return table_field, key2field, columns


__TABULAR_PERIODS__ = {
    '1m': 60000,
    '5m': 300000,
    '15m': 900000,
    '30m': 1800000,
    '60m': 3600000,
    '1h': 3600000,
    '1d': 86400000,
    '1w': 604800000,
    '1mon': 2592000000,
    '1q': 7776000000,
    '1hy': 15552000000,
    '1y': 31536000000,
}


def get_tabular_data(
        codes: list,
        fields: list,
        period: str,
        start_time: str,
        end_time: str,
        count: int = -1,
        **kwargs
):
    from .. import xtbson, xtdata
    import pandas as pd
    import os

    time_format = None
    if period in ('1m', '5m', '15m', '30m', '60m', '1h'):
        time_format = '%Y-%m-%d %H:%M:%S'
    if period in ('1d', '1w', '1mon', '1q', '1hy', '1y'):
        time_format = '%Y-%m-%d'

    if not time_format:
        raise Exception('Unsupported period')

    int_period = __TABULAR_PERIODS__[period]

    table_field, key2field, ori_columns = parse_request_from_fields(fields)

    dfs = []
    client = xtdata.get_client()
    for metaid, keys in table_field.items():
        data_path_dict = xtdata._get_data_file_path(codes, (metaid, int_period))

        for code, file_path in data_path_dict.items():
            if not file_path:
                continue

            if metaid not in key2field:
                continue

            if not os.path.exists(file_path):
                continue

            bson_datas = xtbson.decode(client.read_local_data(file_path, start_time, end_time, count)).get('result')
            df = pd.DataFrame(bson_datas)
            if df.empty:
                continue

            # 移除结果中多余的字段
            drop_columns = [c for c in df.columns if c not in keys]
            if drop_columns:
                df = df.drop(drop_columns, axis=1)
            # 补充请求的字段
            default_null_columns = [c for c in keys if c not in df.columns]
            for c in default_null_columns:
                df.loc[:, c] = keys[c]

            df.rename(columns=key2field[metaid], inplace=True)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    result = result[ori_columns]

    return result
