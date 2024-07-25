/////////////////////////////////////////////////////////////////////////
///��Ʒ HTS
///��˾ ������������ɷ����޹�˾
///�ļ� itpdk_dict.h
///��; �����˲����ֵ�
///�汾
///20210910  5.1.0.0	֣����		�����汾˵��
/////////////////////////////////////////////////////////////////////////

#pragma once


#define PRODUCT             "itpdk"
#define VERSION             "5230"


//������    200 - 600
#define     ERR_CONN_INVALID            -201            //������Ч
#define     ERR_ALLOCSESS_FAILED        -202            //����Ựʧ��
#define     ERR_NOT_LOGIN               -203            //�ͻ�δ��¼
#define     ERR_REQ_FAILED              -204            //��������ʧ��
#define     ERR_WRONG_ANS               -205            //���յ�����Ӧ������
#define     ERR_READ_CONFIG_FILE        -206            //��ȡ�����ļ�ʧ��

#define     ERR_UNKNOW                  -220            //δ֪����
#define     ERR_PARAM_INVALID           -221            //�Ƿ�����
#define     ERR_NOT_SUPPORT             -222            //�ӿڲ�֧�ֵĲ���
#define     ERR_SYSTEM_TYPE_INVALID     -223            //��Чϵͳ����
#define     ERR_NOT_FOUND_CONN_INFO     -224            //�޷���ȡ��������Ϣ
#define     ERR_NO_TRADE                -225            //��ֹ����
#define     ERR_NO_SPECIFY_NODEID       -226            //δָ�����׽ڵ�
#define     ERR_NO_NODEID_INFO          -227            //�޿ͻ��ڵ���Ϣ


//����������
const char svrtype_quote[] = "QUOTE";       //���������
const char svrtype_hts[] = "HTS";           //HTS���׷�����
//ϵͳ����
#define SYSTEM_TYPE_SEC       4           //�ֻ�ϵͳ
#define SYSTEM_TYPE_MAR       7           //����ϵͳ
#define SYSTEM_TYPE_SOP       9           //��Ȩϵͳ
//��ȫ��֤��ʽ
#define FIX_AUTH_STYLE_DTKL   2           //��̬����
#define FIX_AUTH_STYLE_CERT   5           //����֤��
//վ���ַ����
#define FIX_NODE_TYPE_MAC     1           //MAC��ַ
#define FIX_NODE_TYPE_HDSN    2           //Ӳ�����к�
#define SEPARATOR_SVRADDR     ';'         //����м����ַ�ָ���
#define MAX_ERRMSG_SIZE       1024        //��������Ϣ�ַ�����󳤶�
#define MAX_RESULT_SIZE       128         //�ɹ�ʱ���ص��ַ���������󳤶�
//��֤ҵ�����
#define YWLB_YHTOZQ           1           //����ת֤ȯ
#define YWLB_ZQTOYH           2           //֤ȯת����
#define YWLB_CXYHYE           6           //���������
//�ӿ�������Ϣ����
#define NOTIFY_CONNECT        1           //ͨѶ����(ProfileKey,Message)
#define NOTIFY_DISCONNECT     2           //ͨѶ����(ProfileKey,Message)
#define NOTIFY_LOGIN          3           //�ͻ���¼(ProfileKey,AccountId,Message)
#define NOTIFY_FILEUPGRADE    4           //�ļ�����(ProfileKey,Message)
#define NOTIFY_FILEUSING      5           //�ļ�����-�ļ���ռ��
#define NOTIFY_SUBSCIBE       6           //���Ĵ���(ProfileKey,Message)
#define NOTIFY_QUOTESUBSCIBE  7           //���鶩��(Message)
#define NOTIFY_PUSH_ORDER     8           //ί������(AccountId,OrderId,Message)
#define NOTIFY_PUSH_WITHDRAW  9           //��������(AccountId,OrderId,Message)
#define NOTIFY_PUSH_MATCH     10          //�ɽ�����(AccountId,OrderId,Message)
#define NOTIFY_PUSH_INVALID   11          //�ϵ�����(AccountId,OrderId,Message)
#define NOTIFY_PUSH_MODIFYPWD 12          //�ͻ��������ط��޸�����
#define NOTIFY_PUSH_BULLETIN  13          //����--����
#define NOTIFY_PUSH_MESSAGE   14          //��Ϣ--����
#define NOTIFY_ASSETVARY      20          //�ʲ��䶯(AccountId,FundAccount,Message)--�Ƚ�Ƶ������������Ϊ��ʾ��Ϣ��
#define NOTIFY_QUOTEVARY      21          //����䶯({Market,StockCode,varys}) varsΪ����䶯��־��λ���
                                          //         ���磺SH,600600,00000006����ʾSH600600�����¼ۺͳɽ������б䶯
                                          //����䶯���ͷǳ�Ƶ������������Ϊ��ʾ��Ϣ��
#define NOTIFY_BANKTRANSFER   22          //����ת�˽��(ProfileKey,AccountId,Message)
#define NOTIFY_PUSH_DEBETS_CHANGE   23          //��ծ�䶯����
#define NOTIFY_PUSH_QUOTE     24          //˫�߱���ί��ȷ������(AccountId,OrderId,Message)
#define NOTIFY_CONNEVENT_MGR  25          //��������
#define NOTIFY_CONNEVENT_TRADE  26        //��������
#define NOTIFY_CONNEVENT_QUERY  27        //��ѯ����
#define NOTIFY_CONNEVENT_SUBS   28        //��������
//�������
#define JYLB_BUY              1           //����
#define JYLB_SALE             2           //����
#define JYLB_PGJK             3           //��ɽɿ�
#define JYLB_HGRZ             4           //�ع�����
#define JYLB_HGRQ             5           //�ع���ȯ
#define JYLB_BONUS            6           //����
#define JYLB_ZTG              7           //����ת�й�
#define JYLB_XGPH             8           //�¹����
#define JYLB_ZDJY             9           //�Ϻ�ָ������
#define JYLB_CZJY             10          //�Ϻ���ָ����
#define JYLB_ZZZG             11          //תծת��
#define JYLB_ZZHS             12          //תծ����
#define JYLB_ZZSH             13          //תծ���
#define JYLB_PSSG             14          //�����깺
#define JYLB_RGFX             15          //�Ϲ�����
#define JYLB_SG               16          //�͹�
#define JYLB_DF               17          //�Ҹ�
#define JYLB_PSFQ             24          //���۷���
#define JYLB_ETFSG            29          //ETF�깺
#define JYLB_ETFSH            30          //ETF���
#define JYLB_ZYQRK            37          //��Ѻȯ���
#define JYLB_ZYQCK            38          //��Ѻȯ����
#define JYLB_JJRG             41          //�����Ϲ�
#define JYLB_JJSG             42          //�����깺
#define JYLB_JJSH             43          //�������
#define JYLB_JJFHSZ           44          //����ֺ�����
#define JYLB_JJZH             46          //����ת��
#define JYLB_JJFC             47          //����ֲ�
#define JYLB_JJHB             48          //����ϲ�
#define JYLB_YXMR             55          //��������
#define JYLB_YXMC             56          //��������
#define JYLB_DJMR             57          //��������
#define JYLB_DJMC             58          //��������
#define JYLB_MRQR             59          //����ȷ��
#define JYLB_MCQR             60          //����ȷ��
#define JYLB_PHMR             78          //�̺�����
#define JYLB_PHMC             79          //�̺�����
#define JYLB_RZMR             61          //��������
#define JYLB_MQHK             62          //��ȯ����
#define JYLB_MQHQ             63          //��ȯ��ȯ
#define JYLB_RQMC             64          //��ȯ����
#define JYLB_DBWTJ            65          //��������
#define JYLB_DBWFH            66          //��������
#define JYLB_QYHR             67          //ȯԴ����
#define JYLB_QYHC             68          //ȯԴ����
#define JYLB_YQHZ             69          //��ȯ��ת
#define JYLB_HQHZ             70          //��ȯ��ת
#define JYLB_RZQP             71          //����ǿƽ
#define JYLB_RQQP             72          //��ȯǿƽ
#define JYLB_ZJHK             73          //ֱ�ӻ���
#define JYLB_YSYY             76          //Ԥ��ҪԼ
#define JYLB_JCYS             77          //���Ԥ��ҪԼ
#define JYLB_ZFMR			  101		  //��������
#define JYLB_PHDJMR           178         //�̺󶨼�����
#define JYLB_PHDJMC           179         //�̺󶨼�����
#define JYLB_HSCX             212         //���۳���
#define JYLB_SHJJRG           241         //�����Ϲ�
#define JYLB_SHJJSG           242         //�����깺
#define JYLB_SHJJSH           243         //�������
#define JYLB_SHJJFHSZ         244         //����ֺ�����
#define JYLB_SHJJZH           246         //����ת��
#define JYLB_SHJJFC           247         //����ֲ�
#define JYLB_SHJJHB           248         //����ϲ�
//��������
#define DDLX_XJWT                  0           //�޼�
// �Ϻ��м�
#define DDLX_SHHB_ZYWDSYCX         1           //�����嵵��ʱ�ɽ�ʣ�೷��
#define DDLX_SHHB_ZYWDSYZXJ        2           //�����嵵��ʱ�ɽ�ʣ��ת�޼�
#define DDLX_SHHB_DSFZYJ           4           //���ַ����ż۸�
#define DDLX_SHHB_BFZYJ            5           //�������ż۸�
#define DDLX_SHHB_PHDJDZJYSPJ      6           //�̺󶨼۴��ڽ������̼�
// �����м�
#define DDLX_SZSB_DSFZYJ           101         //���ַ����ż۸�
#define DDLX_SZSB_BFZYJ            102         //�������ż۸�
#define DDLX_SZSB_SYCX             103         //��ʱ�ɽ�ʣ�೷��
#define DDLX_SZSB_ZYWDSYCX         104         //�����嵵��ʱ�ɽ�ʣ�೷��
#define DDLX_SZSB_QECJCX           105         //ȫ��ɽ�����
#define DDLX_SZSB_PHDJDZJYSPJ      106         //�̺󶨼۴��ڽ������̼�
#define DDLX_SZSB_PHDJDZJYSJ       107         //�̺󶨼۴��ڽ���ƽ����

//�Ϻ���Ȩ��������
#define DDLX_SHQQ_XJGFD         0           //�޼�GFD
#define DDLX_SHQQ_SJIOC         1           //�м�IOC
#define DDLX_SHQQ_SJZXJGFD      2           //�м�ʣ��ת�޼�GFD
#define DDLX_SHQQ_XJFOK         4           //�޼�FOK
#define DDLX_SHQQ_SJFOK         5           //�м�FOK
//������Ȩ��������
#define DDLX_SZQQ_XJGFD         0           //�޼�ί��
#define DDLX_SZQQ_XJFOK         4           //�޼�ȫ��ɽ�����
#define DDLX_SZQQ_DSFZYJ        101         //���ַ�����ʣ��ת�޼�
#define DDLX_SZQQ_BFZYJ         102         //��������
#define DDLX_SZQQ_SYCX          103         //�м������ɽ�ʣ�೷��
#define DDLX_SZQQ_ZYWDSYCX      104         //�м������嵵ȫ��ɽ�ʣ�೷��
#define DDLX_SZQQ_QECJCX        105         //�м�ȫ��ɽ�����

//������������
#define OT_ALO                1           //�����޼��� At-auction Limit Order(Pre-opening Session and Closing Auction Session)
#define OT_ELO                2           //��ǿ�޼��� Enhanced Limit Order(Continuous Trading Session)
//����仯��־
#define QV_TPBZ               0x00000001  //ͣ�Ʊ�־
#define QV_ZXJ                0x00000002  //���¼�
#define QV_ZSP                0x00000004  //������
#define QV_CJSL               0x00000008  //�ɽ�����(�ɽ����)
#define QV_CCL                0x00000010  //�ֲ���
#define QV_ZGZDJ              0x00000020  //�����ͼ�
#define QV_MRJG1              0x00000100  //����۸�1
#define QV_MCJG1              0x00000200  //�����۸�1
#define QV_MRSL1              0x00000400  //��������1
#define QV_MCSL1              0x00000800  //��������1
#define QV_MRJG2              0x00001000
#define QV_MCJG2              0x00002000
#define QV_MRSL2              0x00004000
#define QV_MCSL2              0x00008000
#define QV_MRJG3              0x00010000
#define QV_MCJG3              0x00020000
#define QV_MRSL3              0x00040000
#define QV_MCSL3              0x00080000
#define QV_MRJG4              0x00100000
#define QV_MCJG4              0x00200000
#define QV_MRSL4              0x00400000
#define QV_MCSL4              0x00800000
#define QV_MRJG5              0x01000000
#define QV_MCJG5              0x02000000
#define QV_MRSL5              0x04000000
#define QV_MCSL5              0x08000000
//��ѯ�ṹ����ʽ
#define SORT_TYPE_DESC        0        //����
#define SORT_TYPE_AES         1        //����

//�걨�������
#define  SBJG_WAITING                0       //����
#define  SBJG_SENDING                1       //����
#define  SBJG_CONFIRM                2       //�ѱ�
#define  SBJG_INVALID                3       //�Ƿ�ί��
#define  SBJG_FUNDREQ                4       //�ʽ�������
#define  SBJG_PARTTRADE              5       //���ֳɽ�
#define  SBJG_COMPLETE               6       //ȫ���ɽ�
#define  SBJG_PTADPWTD               7       //���ɲ���
#define  SBJG_WITHDRAW               8       //ȫ������
#define  SBJG_WTDFAIL                9       //����δ��
#define  SBJG_MANUAL                 10      //�ȴ��˹��걨

// �첽�ص�������������
//      ��Ȩ����
#define FUNC_CALLBACK_KPJY                  0         //��ƽ����
#define FUNC_CALLBACK_KPJY_WITHDRAW         1         //��ƽ���׳���
#define FUNC_CALLBACK_ORDERPETRUST_BJ       2       //����
#define FUNC_CALLBACK_KPJY_WITHDRAW_ALL     3         //��ƽ����ȫ������
#define FUNC_CALLBACK_BJ_WITHDRAW_ALL       4         //����ȫ������
//      �ֻ�����
#define FUNC_CALLBACK_PTMM          30       //��ͨ����
#define FUNC_CALLBACK_ETFSG         31       //ETF�깺
#define FUNC_CALLBACK_ETFSH         32       //ETF���
#define FUNC_CALLBACK_ZQHG          33       //ծȯ�ع�
#define FUNC_CALLBACK_ZQCRK         34       //ծȯ�����
#define FUNC_CALLBACK_LOF           35       //LOF����
#define FUNC_CALLBACK_PHDJ          36       //�̺󶨼�����
#define FUNC_CALLBACK_FXYW          37       //����ҵ��
#define FUNC_CALLBACK_FJY           38       //�ǽ���ҵ��
#define FUNC_CALLBACK_WD_PTMM       40       //��ͨ��������
#define FUNC_CALLBACK_WD_BATCH      41       //������������
#define FUNC_CALLBACK_WD_ETF        42       //ETF����
#define FUNC_CALLBACK_WD_ZQ         43       //ծȯ�ع�����
#define FUNC_CALLBACK_WD_LOF        44       //LOF����
#define FUNC_CALLBACK_WD_PHDJ       45       //�̺󶨼۳���
#define FUNC_CALLBACK_WD_FXYW       46       //����ҵ�񳷵�

//      ��������
#define MR_FUNC_CALLBACK_XYMM          100       //��������
#define MR_FUNC_CALLBACK_RZRQ          101       //������ȯ
#define MR_FUNC_CALLBACK_CHYW          102       //����ҵ��(��ȯ�����ȯ��ȯ)
#define MR_FUNC_CALLBACK_FJYYW         103       //�ǽ���ҵ��(�������ύ�������ﷵ������ȯ��ת)
#define MR_FUNC_CALLBACK_FXYW          104       //����ҵ��(�����깺����ɽɿ�)
#define MR_FUNC_CALLBACK_PHDJ          105       //�̺󶨼�
#define MR_FUNC_CALLBACK_WD_XYMM       120       //������������
#define MR_FUNC_CALLBACK_WD_RZRQ       121       //������ȯ����
#define MR_FUNC_CALLBACK_WD_CHYW       122       //����ҵ�񳷵�(��ȯ�����ȯ��ȯ)
#define MR_FUNC_CALLBACK_WD_FJYYW      123       //�ǽ���ҵ�񳷵�(�������ύ�������ﷵ������ȯ��ת)
#define MR_FUNC_CALLBACK_WD_FXYW       124       //����ҵ�񳷵�(�����깺����ɽɿ�)
#define MR_FUNC_CALLBACK_WD_PHDJ       125       //�̺󶨼۳���

//����--������������
const char table_khh[] = "tKHH";      //�ͻ���
const char table_bz[] = "tBZ";       //����
const char table_jys[] = "tJYS";      //������
const char table_xtdm[] = "tXTDM";     //ϵͳ����
const char table_zqjysx[] = "tZQJYSX";   //֤ȯ��������
const char table_jylb[] = "tJYLB";     //�������
const char table_jgdm[] = "tJGDM";     //��������
const char table_yhcs[] = "tYHCS";     //���в���
const char table_etfxx[] = "tETFXX";    //ETF��Ϣ
const char table_etfmx[] = "tETFMX";    //ETF�ɷֹ���ϸ
const char table_jjhq[] = "tJJHQ";     //������Ϣ
const char table_fjjj[] = "tFJJJ";     //�ּ�����
const char table_xgsg[] = "tXGSG";     //�����¹��깺
const char table_hlcs[] = "tHLCS";     //���ʲ���
const char table_zqhq[] = "tZQHQ";     //֤ȯ����
const char table_jyjw[] = "tJYJW";     //���׼�λ
const char table_zjzh[] = "tZJZH";     //�ʽ��˺�
const char table_zcxx[] = "tZCXX";     //�ʲ�ͳ��
const char table_gdh[] = "tGDH";      //�ɶ���
const char table_yhzh[] = "tYHZH";     //�����˺�
const char table_zqgl[] = "tZQGL";     //֤ȯ�ֲ�
const char table_drwt[] = "tDRWT";     //����ί��
const char table_sscj[] = "tSSCJ";     //ʵʱ�ɽ�
const char table_zzsq[] = "tZZSQ";     //֤ȯ�����ת������
const char table_jgmx[] = "tJGMX";     //������ˮ
const char table_wtls[] = "tWTLS";     //��ʷί��
const char table_zjls[] = "tZJLS";     //�ʽ���ˮ
const char table_zqjk[] = "tZQJK";     //��ǩ�ɿ�
const char table_sodrwt[] = "tSODRWT";   //��Ȩ����ί��
const char table_sohycc[] = "tSOHYCC";   //��Ȩ��Լ�ֲ�
const char table_sobdzqcc[] = "tSOBDZQCC";   //��Ȩ����֤ȯ�ֲ�
const char table_sozhclcc[] = "tSOZHCLCC";   //��Ȩ��ϲ��Գֲ�
const char table_sobzjxx[] = "tSOBZJXX";  //��Ȩ��֤��
const char table_sosscj[] = "tSOSSCJ";     //��Ȩʵʱ�ɽ�
const char table_sofbcj[] = "tSOFBCJ";     //��Ȩ�ֱʳɽ�
const char table_sohydm[] = "tSOHYDM";     //��Ȩ��Լ����
const char table_sohyzh[] = "tSOHYZH";     //��Ȩ��Լ�˻�
const char table_sokhjykzxx[] = "tSOKHJYKZXX";     //��Ȩ�ͻ����׿�����Ϣ
const char table_sokhxqdjszjqk[] = "tSOKHXQDJSZJQK";     //��Ȩ�ͻ���Ȩ�������ʽ�ȱ��
const char table_sokhxqdjszqqk[] = "tSOKHXQDJSZQQK";     //��Ȩ�ͻ���Ȩ������֤ȯȱ��
const char table_sokhxqzpxx[] = "tSOKHXQZPXX";     //��Ȩ�ͻ���Ȩָ����Ϣ
const char table_sokhbdzqbz[] = "tSOKHBDZQBZ";     //��Ȩ�ͻ�����֤ȯ����
const char table_sokhbcxx[] = "tSOKHBCXX";     //��Ȩ�ͻ�������Ϣ

//������ر�
const char table_xyxgsg[] = "tXYXGSG";   //���ñ����¹��깺
const char table_xygdh[] = "tXYGDH";    //���ùɶ���
const char table_xyzjzh[] = "tXYZJZH";   //�����ʽ��˺�
const char table_xyzqgl[] = "tXYZQGL";   //����֤ȯ�ֲ�
const char table_xydrwt[] = "tXYDRWT";   //���õ���ί��
const char table_xysscj[] = "tXYSSCJ";   //����ʵʱ�ɽ�
const char table_xyzc[] = "tXYZC";     //�����ʲ�
const char table_xyfz[] = "tXYFZ";     //���ø�ծ
const char table_xypsqy[] = "tXYPSQY";   //��������Ȩ��
const char table_zgzq[] = "tZGZQ";     //�ʸ�֤ȯ
const char table_rqzq[] = "tRQZQ";     //���֤ȯ����ȯ���
const char table_xydrbd[] = "tXYDRBD";   //�������ø�ծ�䶯��ϸ
const char table_xyrqfzhz[] = "tXYRQFZHZ";   //����������ȯ��ծ����
const char table_xyfzls[] = "tXYFZLS";   //���ø�ծ�䶯��ϸ
const char table_xyzqdm[] = "tXYZQDM";     //����֤ȯ����

//������
const char dict_jys_dl[] = "DL";        //����
const char dict_jys_sq[] = "SQ";        //����
const char dict_jys_zj[] = "ZJ";        //�н�
const char dict_jys_zz[] = "ZZ";        //֣��
const char dict_jys_sn[] = "SN";        //ԭ��
const char dict_jys_sh[] = "SH";        //�Ϻ�
const char dict_jys_sz[] = "SZ";        //����
const char dict_jys_hk_sh[] = "HK";        //����
const char dict_jys_hk_sz[] = "SK";        //���
//����
const char dict_bz_rmb[] = "RMB";       //�����
const char dict_bz_hkd[] = "HKD";       //�۱�
const char dict_bz_usd[] = "USD";       //��Ԫ

