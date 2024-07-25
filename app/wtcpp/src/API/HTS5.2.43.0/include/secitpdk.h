#pragma once


#include <map>
#include <vector>
#include <string>
using std::string;
using std::map;
using std::vector;


#include "itpdk/itpdk_typedef.h"
#include "itpdk/itpdk_dict.h"
#include "secitpdk/secitpdk_struct.h"


//////////////////////////////////////////////////////////////////////////
//��ȡʱ������뼶����linux��winͬGetTickCount64��
ITPDK_API int64 SECITPDK_GetTickCount();
//��ȡʱ���΢�뼶����linux��winͬGetTickCount64��
ITPDK_API int64 SECITPDK_GetDoubleTickCount();
//��ȡ��������������
ITPDK_API int64 SECITPDK_GetTradeDate();
//��ȡ������ϵͳ����
ITPDK_API int64 SECITPDK_GetSystemDate();
//��ȡ��������ʱ��(�����׷�����ʱ������)
ITPDK_API int64 SECITPDK_GetReviseTimeAsLong();
//��ȡ��������ʱ��(�����׷�����ʱ������)
ITPDK_API void SECITPDK_GetReviseTime(char* buffer);
//��ȡ��ǰ��汾��
ITPDK_API void SECITPDK_GetVersion(char* buffer);

//////////////////////////////////////////////////////////////////////////
//��������־·����SECITPDK_Init ����ǰ������Ч
ITPDK_API void SECITPDK_SetLogPath(const char* path);
//���������ļ�����·����SECITPDK_Init ����ǰ������Ч
ITPDK_API void SECITPDK_SetProfilePath(const char* path);
// �����Ƿ�����ITPDK�Զ��������б����ܣ�SECITPDK_Init ����ǰ������Ч
ITPDK_API void SECITPDK_SetAutoReconnect(bool breconnect);
//�����Ƿ��¼��־
ITPDK_API void SECITPDK_SetWriteLog(bool bLog);
//�����Ƿ��¼Fix��־
ITPDK_API void SECITPDK_SetFixWriteLog(bool bLog);
//��������ʱʱ�䣬��λ��
ITPDK_API void SECITPDK_SetTimeOut(int seconds);
//�����Ƿ�����ͨѶ����ѹ����Ĭ�ϲ����ã�Ҫע�����㣺1.��������»�Ӱ�쵽���� 2.��̨�м���汾��68���µ����ã����ܵ��¿ͻ�������Ӧ��
ITPDK_API void SECITPDK_SetZip(bool bZip);
//�����Ƿ�����ͨѶ���ݼ��ܣ�Ĭ�ϲ����ã�ע�⣺��������»�Ӱ�쵽���ܣ�
ITPDK_API void SECITPDK_SetCrypt(bool bCrypt);

//��ʼ��
//��������Ϊ��ǰͷ�ļ��汾�ţ�ֱ��ȡ�궨��HEADER_VER�����ڲ�У��ͷ�ļ��Ϳ��ļ��汾�Ƿ�ƥ�䣬��0��У��
ITPDK_API bool SECITPDK_Init(int version);
//�˳�
ITPDK_API void SECITPDK_Exit();
//��ȡ���������Ϣ�ʹ�����
ITPDK_API int64 SECITPDK_GetLastError(char* result_msg);
//�������´�����Ϣ
ITPDK_API void SECITPDK_SetLastError(int64 result_code, const char* result_msg, const char* result_text = nullptr, ITPDK_CusReqInfo* cusreqinfo = nullptr);

/*
    ������Ϣ�ص�����
        ��Ϣ����pMsgΪjson��

        ������Ϣ -- NOTIFY_PUSH_WITHDRAW
        �ϵ���Ϣ -- NOTIFY_PUSH_INVALID
        ȷ����Ϣ -- NOTIFY_PUSH_ORDER
            WTH     int64       ί�к�
            KHH     char[]      �ͻ���
            YYB     char[]      Ӫҵ��
            GDH     char[]      �ɶ���
            JYS     char[]      ������
            ZQDM    char[]      ֤ȯ����
            XWH     char[]      ϯλ��
            BZ      char[]      ����
            SBJG    int64       �걨���
            JYLB    int64       �������
            CXBZ    char        ������־
            CXWTH   int64       ����ί�к�
            CJBH    char[]      �ɽ����
            CJSJ    char[]      �ɽ�ʱ��
            CJSL    int64       �ɽ�����
            CJJG    double      �ɽ��۸�
            CJJE    doube       �ɽ����
        �ɽ���Ϣ -- NOTIFY_PUSH_MATCH
            KHH     char[]      �ͻ���
            JYS     char[]      ������
            ZQDM    char[]      ֤ȯ����
            WTH     int64       ί�к�
            CJSL    int64       �ɽ�����
            CJJG    double      �ɽ��۸�
        �ʲ��䶯��Ϣ -- NOTIFY_ASSETVARY
            KHH     char[]      �ͻ���
            ZJZH    char[]      �ʽ��˺�
            ZZC     double      ���ʲ�
            KYZJ    double      �����ʽ�
            ZXSZ    double      ������ֵ

    func        [in]        �ص�����
*/
ITPDK_API void SECITPDK_SetMsgCallback(pMessageCallbackMethod func);

//���ýṹ����Ϣ�ص�����
ITPDK_API void SECITPDK_SetStructMsgCallback(pStructMessageCallbackMethod func);

// �첽�ص�����
ITPDK_API void SECITPDK_SetFuncCallback(pAsyncCallbackFunc func);

// ���ӵ��ߡ��ָ��¼��ص�����
/*
    pConnEventCallback

    pKhh        [in]        �ͻ���
    pConnKey    [in]        ����������
    nEvent      [in]        �¼�:��0���ָ���1���Ͽ�
*/
ITPDK_API void SECITPDK_SetConnEventCallback(pConnEventCallback func);

// ����վ����Ϣ
ITPDK_API bool SECITPDK_SetNode(const char* val);
// ������վ����Ϣ
ITPDK_API bool SECITPDK_CheckCZZD(const char* lpKeyName, const char* khh, const char* lpCZZD);

// ����ί�з�ʽ
ITPDK_API bool SECITPDK_SetWTFS(const char* val);

/*
    ��ѯ�ͻ��ڵ���Ϣ

    lpKeyName   [in]        ʹ�õ�����Keyֵ
    khh         [in]        �ͻ���
    return                �ɹ����ؽ��������ʧ�ܷ���<0
*/
ITPDK_API int SECITPDK_QueryCusNodeinfo(const char* lpKeyName, const char* khh, vector<ITPDK_KHJD>& arKHJD);

/*
    �����û����׽ڵ�

    khh         [in]        �ͻ���
    nodeid      [in]        �ڵ��
*/
ITPDK_API bool SECITPDK_SetCusNodeID(const char* khh, int nodeid);

/*
    ���������Ƿ���Ч
    lpKeyName   [in]        ʹ�õ�����Keyֵ
    return                  ��Ч����>0��ʧ�ܷ���<=0
*/
ITPDK_API bool SECITPDK_IsValidConnkey(const char* lpKeyName);

/*
    ���׵�¼
    lpKeyName   [in]        ʹ�õ�����Keyֵ
    khh         [in]        �ͻ���
    pwd         [in]        ��������
    return              �ɹ�����>0��ʧ�ܷ���<=0
*/
ITPDK_API int64 SECITPDK_TradeLogin(const char* lpKeyName, const char* khh, const char* pwd);

/*
    ���׵�¼
    lpKeyName   [in]        ʹ�õ�����Keyֵ
    cusreqinfo  [in/out]    �ͻ���Ϣ
    return              �ɹ�����>0��ʧ�ܷ���<=0
*/
ITPDK_API int64 SECITPDK_TradeLoginEx(const char* lpKeyName, ITPDK_CusReqInfo& cusreqinfo);

/*
    �ǳ�
    khh         [in]        �ͻ���
*/
ITPDK_API int64 SECITPDK_TradeLogout(const char* khh);

/*
    �����û�token

    khh         [in]        �ͻ���
    token       [in]        ����
    return              �ɹ�����true��ʧ�ܷ���false
*/
ITPDK_API bool SECITPDK_SetCusToken(const char* khh, int64 token);

/*
    ��ѯ�ֵ�����

    tablename   [in]        ����
    khh         [in]        �ͻ���
    nRowcount   [in]        ���ؼ�¼����0ʱ����ȫ����
    nBrowindex  [in]        ��ҳ��0ʱ��ͷ�飩
    return              ���ز�ѯ���ļ�¼��
*/
ITPDK_API int64 SECITPDK_BeginQuery(const char* tablename, const char* khh, int nRowcount = 0, int nBrowindex = 0);

/*
    ��ȡ��ѯ����
        ���ò�ѯ�ӿ�(SECITPDK_BeginQuery��)��ͨ���ýӿڻ�ȡ����

    tablename   [in]        ����
    khh         [in]        �ͻ���
    buffer      [out]        ����buffer�����ڷ�������
    buffsize    [in]        buffer��С
    nRecno      [in]        ���ݱ��
    return              �ɹ�����true��ʧ��false
*/
ITPDK_API bool SECITPDK_GetQuery(const char* tablename, const char* khh, void* buffer, int buffsize, int nRecno);

/*
    �ͷŲ�ѯ����
        ���ò�ѯ�ӿں󣬲���ʹ�ò�ѯ����ʱ�����øýӿ��ͷŲ�ѯ����

    tablename   [in]        ����
    khh         [in]        �ͻ���
    return              �ɹ�����true��ʧ��false
*/
ITPDK_API bool SECITPDK_EndQuery(const char* tablename, const char* khh);

/*
    ��ѯ�ͻ���Ϣ

    sKhh        [in]        �ͻ���
    return              �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryKHXX(const char* sKhh, ITPDK_KHH& khhRecord);

/*
    ��ѯ�ʽ���Ϣ

    khh         [in]        �ͻ���
    arZjzh      [out]        �������
    return              ���ز�ѯ���ļ�¼��
*/
ITPDK_API int64 SECITPDK_QueryFundInfo(const char* khh, vector<ITPDK_ZJZH>& arZjzh);
ITPDK_API int64 SECITPDK_QueryFundInfoEx(ITPDK_CusReqInfo& cusreqinfo, vector<ITPDK_ZJZH>& arZjzh);

/*
    ��ѯ�ɶ���Ϣ

    khh         [in]        �ͻ���
    arZjzh      [out]        �������
    return              ���ز�ѯ���ļ�¼��
*/
ITPDK_API int64 SECITPDK_QueryAccInfo(const char* khh, vector<ITPDK_GDH>& arGdh);
ITPDK_API int64 SECITPDK_QueryAccInfoEx(ITPDK_CusReqInfo& cusreqinfo, vector<ITPDK_GDH>& arGdh);

/*
    ��ѯ����ί��

    khh         [in]        �ͻ���
    nType       [in]        �Ƿ����ѯ�ɳ�ί�У�0��ѯȫ����1��ѯ�ɳ���
    nSortType   [in]        ����ʽ��0����1����
    nRowcount   [in]        ��������
    nBrowindex  [in]        ��ҳ����ί�кţ�
    jys         [in]        ����������֤ȯ�������ʹ�ã�������������ָ��ʱ����ѯָ��֤ȯ���롢��������ί��
    zqdm        [in]        ֤ȯ���룬�뽻�������ʹ�ã�������������ָ��ʱ����ѯָ��֤ȯ���롢��������ί��
    lWth        [in]        ί�кţ�Ϊ��ʱ��ѯ����ί�кţ�
    arDrwt      [out]        �������
    return              ���ز�ѯ���ļ�¼��
*/
ITPDK_API int64 SECITPDK_QueryOrders(const char* khh, int nType, int nSortType, int nRowcount, int64 nBrowindex, const char* jys, const char* zqdm, int64 lWth, vector<ITPDK_DRWT>& arDrwt);
ITPDK_API int64 SECITPDK_QueryOrdersEx(ITPDK_CusReqInfo& cusreqinfo, int nType, int nSortType, int nRowcount, int64 nBrowindex, const char* jys, const char* zqdm, int64 lWth, vector<ITPDK_DRWT>& arDrwt);

/*
    ��ѯ�ɽ�

    khh         [in]        �ͻ���
    nType       [in]        �Ƿ����ѯCJSL>0���֣�0��ѯȫ����1��ѯCJSL>0���֣�
    nSortType   [in]        ����ʽ��0����1����
    nRowcount   [in]        ��������
    nBrowindex  [in]        ��ҳ�����ɽ���ţ�
    jys         [in]        ��������Ϊ��ʱ��ѯ���н�������
    zqdm        [in]        ֤ȯ���루Ϊ��ʱ��ѯ���д��룩
    lWth        [in]        ί�кţ�Ϊ��ʱ��ѯ����ί�кţ�
    arSscj      [out]        �������
    return              ���ز�ѯ���ļ�¼��
*/
ITPDK_API int64 SECITPDK_QueryMatchs(const char* khh, int nType, int nSortType, int nRowcount, int64 nBrowindex, const char* jys, const char* zqdm, int64 lWth, vector<ITPDK_SSCJ>& arSscj);
ITPDK_API int64 SECITPDK_QueryMatchsEx(ITPDK_CusReqInfo& cusreqinfo, int nType, int nSortType, int nRowcount, int64 nBrowindex, const char* jys, const char* zqdm, int64 lWth, vector<ITPDK_SSCJ>& arSscj);

/*
    ��ѯ�ֲ�

    khh         [in]        �ͻ���
    nSortType   [in]        ����ʽ��0����1����
    nRowcount   [in]        ��������
    nBrowindex  [in]        ��ҳ�����ֱֲ�ţ�
    gdh         [in]        �ɶ��ţ�Ϊ��ʱ��ѯ���йɶ��ţ�
    jys         [in]        ��������Ϊ��ʱ��ѯ���н�������
    zqdm        [in]        ֤ȯ���루Ϊ��ʱ��ѯ���д��룩
    exeflag     [in]        >0����ʵ������ĸ���ӯ��
    arZqgl      [out]        �������
    return              ���ز�ѯ���ļ�¼��
*/
ITPDK_API int64 SECITPDK_QueryPositions(const char* khh, int nSortType, int nRowcount, int64 nBrowindex, const char* gdh, const char* jys, const char* zqdm, int32 exeflag, vector<ITPDK_ZQGL>& arZqgl);
ITPDK_API int64 SECITPDK_QueryPositionsEx(ITPDK_CusReqInfo& cusreqinfo, int nSortType, int nRowcount, int64 nBrowindex, const char* gdh, const char* jys, const char* zqdm, int32 exeflag, vector<ITPDK_ZQGL>& arZqgl);

/*
    ��ѯ�ʽ𶳽���ϸ

    khh         [in]        �ͻ���
    sZjzh       [in]        �ʽ��˺�
    nRowcount   [in]        ��������
    nBrowindex  [in]        ��ҳ������ˮ�ţ�
    nDJLB       [in]        �������Ϊ0ʱ��ѯ���У�
    arZJLS      [out]        �������
    return              ���ز�ѯ���ļ�¼��
*/
ITPDK_API int64 SECITPDK_QueryFreezeDetails(const char* khh, const char* sZjzh, int nRowcount, int64 nBrowindex, int nDJLB, vector<ITPDK_ZJLS>& arZJLS);
ITPDK_API int64 SECITPDK_QueryFreezeDetailsEx(ITPDK_CusReqInfo& cusreqinfo, const char* sZjzh, int nRowcount, int64 nBrowindex, int nDJLB, vector<ITPDK_ZJLS>& arZJLS);

/*
    ��ѯ����Ȩ��

    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpGdh       [in]        �ɶ���
    arPsqy      [out]        �������
    return              �ɹ����ؽ������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryPSQY(const char* lpKhh, const char* lpJys, const char* lpGdh, vector<ITPDK_PSQYInfo>& arPsqy);
ITPDK_API int64 SECITPDK_QueryPSQYEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpGdh, vector<ITPDK_PSQYInfo>& arPsqy);

/*
    ��ѯ֤ȯ������Ϣ

    khh         [in]        �ͻ���
    nRowcount   [in]        ��������
    lpBrowindex [in]        ��ҳ��λ��
    lpJys       [in]        ������
    lpZqdm      [in]        ֤ȯ����
    lpZQLB      [in]        ֤ȯ���
    exeflag     [in]        Ĭ��Ϊ0����Ϊ2ʱ��ѯ�¹ɴ��룬��Ϊ3ʱ��ѯ��ծ���룬��������ͨ��ѯ
    arZQDM      [out]        �������
    return              �ɹ����ؽ������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryZQDMInfo(const char* khh, int nRowcount, const char *lpBrowindex, const char* lpJys, const char* lpZqdm, const char* lpZQLB, int exeflag, vector<ITPDK_ZQDM>& arZQDM);
ITPDK_API int64 SECITPDK_QueryZQDMInfoEx(ITPDK_CusReqInfo& cusreqinfo, int nRowcount, const char* lpBrowindex, const char* lpJys, const char* lpZqdm, const char* lpZQLB, int exeflag, vector<ITPDK_ZQDM>& arZQDM);

/*
    ��ѯ����ͶƱ��Ϣ

    khh         [in]        �ͻ���
    nRowcount   [in]        ��������
    nBrowindex  [in]        ��ҳ��λ��
    lpJys       [in]        ������
    lpZqdm      [in]        ֤ȯ����
    lpZQLB      [in]        ֤ȯ���
    arZQDM      [out]        �������
    return              �ɹ����ؽ������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryTPXX(const char* khh, int nRowcount, int64 nBrowindex, const char* lpJys, const char* lpZqdm, vector<ITPDK_TPXX>& arTPXX);
ITPDK_API int64 SECITPDK_QueryTPXXEx(ITPDK_CusReqInfo& cusreqinfo, int nRowcount, int64 nBrowindex, const char* lpJys, const char* lpZqdm, vector<ITPDK_TPXX>& arTPXX);

/*
    ��ѯETF������Ϣ

    khh         [in]        �ͻ���
    jjdm        [in]        ETF����
    arEtfxx     [out]       �������
    return                �ɹ����ؽ������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryETFXX(const char* khh, const char* jjdm, const char* sgshdm, ITPDK_ETFXX& stEtfxx);
ITPDK_API int64 SECITPDK_QueryETFXXEx(ITPDK_CusReqInfo& cusreqinfo, const char* jjdm, const char* sgshdm, ITPDK_ETFXX& stEtfxx);

/*
    ��ѯETF��ϸ���ɷֹ���Ϣ

    khh         [in]        �ͻ���
    jjdm        [in]        ETF����
    arEtfmx     [out]        �������
    return                �ɹ����ؽ������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryETFShare(const char* khh, const char* jjdm, vector<ITPDK_ETFMX>& arEtfmx);
ITPDK_API int64 SECITPDK_QueryETFShareEx(ITPDK_CusReqInfo& cusreqinfo, const char* jjdm, vector<ITPDK_ETFMX>& arEtfmx);

/*
    ��ѯ�¹��깺��Ϣ
    sJys        [in]        ������
    sZqdm       [in]        ֤ȯ����
    nRgrq       [in]        �Ϲ�����
    sFxfs       [in]        ���з�ʽ
    arXGSG      [out]       �������

    return                �ɹ�����������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryNewStockInfo(const char* sJys, const char* sZqdm, int64 nRgrq, const char* sFxfs, vector<ITPDK_XGSG>& arXGSG);

/*
    ��ѯ�¹���ǩ

    pReqXGZQ    [in]        ��ѯ�¹���ǩ��νṹ��
    arXGZQ      [out]       �������
    return                  �ɹ�����������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryXGZQ(ITPDK_REQ_XGZQ &pReqXGZQ, vector<ITPDK_XGZQ>& arXGZQ);

/*
    ��Ų�ѯ

    pReqPHCX    [in]        ��Ų�ѯ��νṹ��
    arPHCX      [out]       �������
    return                  �ɹ�����������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryPHCX(ITPDK_REQ_PHCX &pReqPHCX, vector<ITPDK_PHCX>& arPHCX);

/*
    ��ѯ��֤ҵ����ˮ

    pReqYZYWLS  [in]        ��ѯ��֤ҵ����ˮ��νṹ��
    arYZYWLS    [out]       �������
    return                  �ɹ�����������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryYZYWLS(ITPDK_REQ_YZYWLS &pReqYZYWLS, vector<ITPDK_YZYWLS>& arYZYWLS);

/*
    ��ѯ�ͻ����д���

    pReqYHDM  [in]          ��ѯ���д�����νṹ��
    arYHDM    [out]         �������
    return                  �ɹ�����������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryYHDM(ITPDK_REQ_YHDM &pReqYHDM, vector<ITPDK_YHDM>& arYHDM);

/*
    ��ѯ�������

    pReqYHYE    [in]        ��ѯ���������νṹ��
    arYHYE      [out]       �������
    return                  �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryYHYE(ITPDK_REQ_YHYE &pReqYHYE, vector<ITPDK_YHYE>& arYHYE);

/*
    ��֤ת��ҵ��

    pReqYZZZ    [in]        ��֤ת����νṹ��
    sSQH        [out]       �����
    return                  �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_BankBusiness(ITPDK_REQ_YZZZ &pReqYZZZ, char *sSQH);

/*
    ��ͨ����
        ע����׺Ex�ӿ�ͨ��cusreqinfo��������ͻ��š��ɶ��š��ڵ�š�token
    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ֤ȯ����
    nJylb       [in]        �������(JYLB_BUY��JYLB_SALE)
    lWtsl       [in]        ί������
    lfWtjg      [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx       [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh       [in]        �ɶ���
    return                �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_OrderEntrust(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_OrderEntrustEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx);

/*
    �첽��ͨ����
        ע����׺Ex�ӿ�ͨ��cusreqinfo��������ͻ��š��ɶ��š��ڵ�š�token
    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ֤ȯ����
    nJylb       [in]        �������(JYLB_BUY��JYLB_SALE)
    lWtsl       [in]        ί������
    lfWtjg      [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx       [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh       [in]        �ɶ���
    nKFSBDBH    [in]        �����̱��ر�ţ���δ�ͣ�����itpdk�Զ�����(�Զ���ί�кţ���ȷ�ϡ��ɽ���������Ϣ�л������ֵ)
    return              �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_OrderEntrust_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_OrderEntrustEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, int64 nKFSBDBH = 0);

/*
    ��ͨ��������ί��

    lpKhh       [in]        �ͻ���
    arBatOrder  [in/out]    ����ί����Ϣ
    nWTPCH      [in]        ί�����κţ�������������������ÿ��ί�е���Ϣ������Ҳ�����
    return                  С��0����ʾ�ǽ��״�������������޷�����ί�еĴ��󣬿�ͨ��SECITPDK_GetLastError��ȡ������Ϣ
                        ��������ʾ����ʧ�ܵ���������ͨ��������ȡʧ����Ϣ
*/
ITPDK_API int64 SECITPDK_BatchOrderEntrust(const char* lpKhh, vector<BatchOrderInfo>& arBatOrder, int64 nWTPCH);
ITPDK_API int64 SECITPDK_BatchOrderEntrustEx(ITPDK_CusReqInfo& cusreqinfo, vector<BatchOrderInfo>& arBatOrder, int64 nWTPCH);

/*
    �첽��ͨ��������ί��

    lpKhh       [in]        �ͻ���
    arBatOrder  [in/out]    ����ί����Ϣ
    nWTPCH      [in]        ί�����κţ�������������������ÿ��ί�е���Ϣ������Ҳ�����
    return              �ɹ�����>0
*/
ITPDK_API int64 SECITPDK_BatchOrderEntrust_ASync(const char* lpKhh, vector<BatchOrderInfo>& arBatOrder, int64 nWTPCH);
ITPDK_API int64 SECITPDK_BatchOrderEntrustEx_ASync(ITPDK_CusReqInfo& cusreqinfo, vector<BatchOrderInfo>& arBatOrder, int64 nWTPCH);

/*
    ETF����

    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ETF����
    nJylb       [in]        �������(JYLB_ETFSG�깺��JYLB_ETFSH���)
    lWtsl       [in]        ί������
    lpGdh       [in]        �ɶ���
    lpStep      [in]        ��չ�ֶ�(Ĭ�Ͽ�)
    return              �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ETFEntrust(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, const char* lpGdh, const char* lpStep = "");
ITPDK_API int64 SECITPDK_ETFEntrustEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, const char* lpStep = "");

/*
    �첽ETF����

    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ETF����
    nJylb       [in]        �������(JYLB_ETFSG�깺��JYLB_ETFSH���)
    lWtsl       [in]        ί������
    lpGdh       [in]        �ɶ���
    lpStep      [in]        ��չ�ֶ�(Ĭ�Ͽ�)
    nKFSBDBH    [in]        �����̱��ر�ţ���δ�ͣ�����itpdk�Զ�����(�Զ���ί�кţ���ȷ�ϡ��ɽ���������Ϣ�л������ֵ)
    return              �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ETFEntrust_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, const char* lpGdh, const char* lpStep = "", int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_ETFEntrustEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, const char* lpStep = "", int64 nKFSBDBH = 0);

/*
    ETF�����ֽ��Ϲ�

    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ETF����
    nJylb       [in]        �������(JYLB_JJRG �����Ϲ�)
    lWtsl       [in]        ί������
    lpGdh       [in]        �ɶ���
    lpStep      [in]        ��չ�ֶ�(Ĭ�Ͽ�)
    return              �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ETFWSXJRG(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, const char* lpGdh, const char* szStep = "");
ITPDK_API int64 SECITPDK_ETFWSXJRGEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, const char* szStep = "");

/*
    �첽ETF�����ֽ��Ϲ�

    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ETF����
    nJylb       [in]        �������(JYLB_ETFWSXJRG ETF�����ֽ��Ϲ�)
    lWtsl       [in]        ί������
    lpGdh       [in]        �ɶ���
    lpStep      [in]        ��չ�ֶ�(Ĭ�Ͽ�)
    nKFSBDBH    [in]        �����̱��ر�ţ���δ�ͣ�����itpdk�Զ�����(�Զ���ί�кţ���ȷ�ϡ��ɽ���������Ϣ�л������ֵ)
    return              �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ETFWSXJRG_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, const char* lpGdh, const char* szStep = "", int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_ETFWSXJRGEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, const char* szStep = "", int64 nKFSBDBH = 0);

/*
    ETF���ӹ�����

    lpKhh           [in]        �ͻ���
    lpJjdm          [in]        ETF����
    nJylb           [in]        �������(JYLB_BUY��JYLB_SALE)
    lWtsl           [in]        ί������
    mpJysGdh        [in]        �������ɶ���
    pSuccRecords    [out]       �����ɹ���¼
    pFailRecords    [out]       ����ʧ�ܼ�¼
    nLSH            [in]        ��ˮ�ţ�������������������ÿ��ί�е���Ϣ������Ҳ�����
    return                  С��0����ʾ�ǽ��״�������������޷�����ί�еĴ��󣬿�ͨ��SECITPDK_GetLastError��ȡ������Ϣ
                            ��������ʾ�������ӹ�ʧ�ܵ���������ͨ��������ȡʧ�ܼ�¼
*/
ITPDK_API int64 SECITPDK_ETFBasketEntrust(const char* lpKhh, const char* lpJjdm, int nJylb, int64 lWtsl, map<string, string>& mpJysGdh, vector<ETFBasket>* pSuccRecords = nullptr, vector<ETFBasket>* pFailRecords = nullptr, int64 nLSH = 0);
ITPDK_API int64 SECITPDK_ETFBasketEntrustEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJjdm, int nJylb, int64 lWtsl, map<string, string>& mpJysGdh, vector<ETFBasket>* pSuccRecords = nullptr, vector<ETFBasket>* pFailRecords = nullptr, int64 nLSH = 0);

/*
    ծȯ����ع�

    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ֤ȯ����
    nJylb       [in]        �������(JYLB_HGRQ��JYLB_HGRZ)
    lWtsl       [in]        ί������
    lfWtjg      [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx       [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh       [in]        �ɶ���
    return              �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ZQHGEntrust(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_ZQHGEntrustEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, double lfWtjg, int64 lDdlx);

/*
    �첽ծȯ����ع�

    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ֤ȯ����
    nJylb       [in]        �������(JYLB_HGRQ��JYLB_HGRZ)
    lWtsl       [in]        ί������
    lfWtjg      [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx       [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh       [in]        �ɶ���
    nKFSBDBH    [in]        �����̱��ر�ţ���δ�ͣ�����itpdk�Զ�����(�Զ���ί�кţ���ȷ�ϡ��ɽ���������Ϣ�л������ֵ)
    return               �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ZQHGEntrust_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_ZQHGEntrustEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, double lfWtjg, int64 lDdlx, int64 nKFSBDBH = 0);

/*
    ծȯ�����

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ��Ѻ����
    nJylb        [in]        �������(JYLB_ZYQRK��JYLB_ZYQCK)
    lWtsl        [in]        ί������
    lpGdh        [in]        �ɶ���
    return              �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ZQHGCrk(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, const char* lpGdh);
ITPDK_API int64 SECITPDK_ZQHGCrkEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl);

/*
    �첽ծȯ�����

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ��Ѻ����
    nJylb        [in]        �������(JYLB_ZYQRK��JYLB_ZYQCK)
    lWtsl        [in]        ί������
    lpGdh        [in]        �ɶ���
    nKFSBDBH     [in]        �����̱��ر�ţ���δ�ͣ�����itpdk�Զ�����(�Զ���ί�кţ���ȷ�ϡ��ɽ���������Ϣ�л������ֵ)
    return              �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ZQHGCrk_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, const char* lpGdh, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_ZQHGCrkEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, int64 nKFSBDBH = 0);

/*
    ����ҵ��

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_PGJK�����ծ�ɿJYLB_PSSG�����깺|ծȯ�䷢��JYLB_BUY�׷��������롢JYLB_PSFQ����ծȯ����)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FXYW(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_FXYWEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx);

/*
    �첽����ҵ��

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_PGJK�����ծ�ɿJYLB_PSSG�����깺|ծȯ�䷢��JYLB_BUY�׷��������롢JYLB_PSFQ����ծȯ����)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FXYW_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_FXYWEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, int64 nKFSBDBH = 0);

/*
    LOF����ҵ��

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_JJRG�����Ϲ�,JYLB_JJSG�����깺,JYLB_JJSH�������,JYLB_JJFC����ֲ�,JYLB_JJHB����ϲ�)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�(�ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_Lof(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, double lfWtjg, int8 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_LofEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, double lfWtjg, int8 lDdlx);

/*
    �첽LOF����ҵ��

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_JJRG�����Ϲ�,JYLB_JJSG�����깺,JYLB_JJSH�������,JYLB_JJFC����ֲ�,JYLB_JJHB����ϲ�)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�(�ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_Lof_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, double lfWtjg, int8 lDdlx, const char* lpGdh, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_LofEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int32 lWtsl, double lfWtjg, int8 lDdlx, int64 nKFSBDBH = 0);

/*
    ���ڽ���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_YXMR��������,JYLB_YXMC��������,JYLB_DJMR��������,JYLB_DJMC��������,JYLB_MRQR����ȷ��,JYLB_MCQR����ȷ��,JYLB_PHMR�̺�����,JYLB_PHMC�̺�����)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    lpDfxw       [in]        �Է�ϯλ
    lpDfgdh      [in]        �Է��ɶ���
    lpLxr        [in]        ��ϵ��
    lpLxdh       [in]        ��ϵ�绰
    lpHyh        [in]        ��Լ��
    nPddm        [in]        ��Գɽ���־��>0 ��ʾ��Գɽ���
    nTdbz        [in]        �ض��ɷݴ��ڽ��ױ�־��>0 ��ʾ�ض��ɷݽ��ף�
    return                �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_DZJY(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh, const char* lpDfxw, const char* lpDfgdh, const char* lpLxr, const char* lpLxdh, const char* lpHyh, int nPddm, int nTdbz);
ITPDK_API int64 SECITPDK_DZJYEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpDfxw, const char* lpDfgdh, const char* lpLxr, const char* lpLxdh, const char* lpHyh, int nPddm, int nTdbz);

/*
    �̺󶨼۽���
        ע����׺Ex�ӿ�ͨ��cusreqinfo��������ͻ��š��ɶ��š��ڵ�š�token
    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_PHDJMR��JYLB_PHDJMC)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_PHDJEntrust(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_PHDJEntrustEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx);

/*
    �첽�̺󶨼۽���
        ע����׺Ex�ӿ�ͨ��cusreqinfo��������ͻ��š��ɶ��š��ڵ�š�token
    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_PHDJMR��JYLB_PHDJMC)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    nKFSBDBH     [in]        �����̱��ر�ţ���δ�ͣ�����itpdk�Զ�����(�Զ���ί�кţ���ȷ�ϡ��ɽ���������Ϣ�л������ֵ)
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_PHDJEntrust_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_PHDJEntrustEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, int64 nKFSBDBH = 0);

/*
    �ǽ���ҵ��

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_HSCX���۳�����JYLB_ZZZGתծת�ɡ�JYLB_ZZHSתծ���ۡ�JYLB_YSYYԤ��ҪԼ��JYLB_JCYS���Ԥ��ҪԼ)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    lpYSYYSGR    [in]        Ԥ��ҪԼ�չ���
    return                �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FJY(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh, const char* lpYSYYSGR);
ITPDK_API int64 SECITPDK_FJYEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpYSYYSGR);

/*
    �첽�ǽ���ҵ��

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_HSCX���۳�����JYLB_ZZZGתծת�ɡ�JYLB_ZZHSתծ���ۡ�JYLB_YSYYԤ��ҪԼ��JYLB_JCYS���Ԥ��ҪԼ)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    lpYSYYSGR    [in]        Ԥ��ҪԼ�չ���
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FJY_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh, const char* lpYSYYSGR, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_FJYEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpYSYYSGR, int64 nKFSBDBH = 0);

/*
    �Ϻ�����ͨҵ��

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_JJRG�����Ϲ���JYLB_JJSG�����깺��JYLB_JJSH������ء�JYLB_JJFHSZ����ֺ����á�JYLB_JJZH����ת����JYLB_ZTG����ת�й�)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�(�ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    nFhfs        [in]        �ֺ췽ʽ(���÷ֺ췽ʽʱ������,0 תͶ,1 �ֽ�
    lpDfjjdm     [in]        Ŀ��������
    lpGdh        [in]        �ɶ���
    return                �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_SHJJT(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, int nFhfs, const char* lpDfjjdm, const char* lpGdh);
ITPDK_API int64 SECITPDK_SHJJTEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, int nFhfs, const char* lpDfjjdm);

/*
    �첽�Ϻ�����ͨҵ��

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_JJRG�����Ϲ���JYLB_JJSG�����깺��JYLB_JJSH������ء�JYLB_JJFHSZ����ֺ����á�JYLB_JJZH����ת����JYLB_ZTG����ת�й�)
    lWtsl        [in]        ί������
    lfWtjg       [in]        ί�м۸�(�ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    nFhfs        [in]        �ֺ췽ʽ(���÷ֺ췽ʽʱ������,0 תͶ,1 �ֽ�
    lpDfjjdm     [in]        Ŀ��������
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_SHJJT_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, int nFhfs, const char* lpDfjjdm, const char* lpGdh, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_SHJJTEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, int nFhfs, const char* lpDfjjdm, int64 nKFSBDBH = 0);

/*
    �۹�ͨ����ί��
        ע����׺Ex�ӿ�ͨ��cusreqinfo��������ͻ��š��ɶ��š��ڵ�š�token
    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpGdh        [in]        �ɶ���
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������
    nWtsl        [in]        ί������
    dWtjg        [in]        ί�м۸�
    nDdlx        [in]        ��������
    nDdjyxz      [in]        ������������
    nDdsxxz      [in]        ����ʱЧ����
    nDdyxrq      [in]        ������Ч����
    dZsxj        [in]        ֹ���޼�
    nDdlb        [in]        �������
    sKZLSH       [in]        ��չ��ˮ��
    sKFSDM       [in]        �����������̴���
    sStep        [in]        STEP��
    nWTPCH       [in]        ί�����κ�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_GGTEntrust(const char* lpKhh, const char* lpJys, const char* lpGdh, const char* lpZqdm, int nJylb, int64 nWtsl, double dWtjg, int nDdlx, int nDdjyxz, int nDdsxxz, int nDdyxrq, double dZsxj, int nDdlb, const char* sKZLSH, const char* sKFSDM, const char* sStep, int64 nWTPCH);
ITPDK_API int64 SECITPDK_GGTEntrustEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 nWtsl, double dWtjg, int nDdlx, int nDdjyxz, int nDdsxxz, int nDdyxrq, double dZsxj, int nDdlb, const char* sKZLSH, const char* sStep, int64 nWTPCH);

/*
    �첽�۹�ͨ����ί��
        ע����׺Ex�ӿ�ͨ��cusreqinfo��������ͻ��š��ɶ��š��ڵ�š�token
    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpGdh        [in]        �ɶ���
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������
    nWtsl        [in]        ί������
    dWtjg        [in]        ί�м۸�
    nDdlx        [in]        ��������
    nDdjyxz      [in]        ������������
    nDdsxxz      [in]        ����ʱЧ����
    nDdyxrq      [in]        ������Ч����
    dZsxj        [in]        ֹ���޼�
    nDdlb        [in]        �������
    sKZLSH       [in]        ��չ��ˮ��
    sKFSDM       [in]        �����������̴���
    nKFSBDBH     [in]        �����������̱��ر��
    sStep        [in]        STEP��
    nWTPCH       [in]        ί�����κ�
    nKFSBDBH     [in]        �����̱��ر�ţ���δ�ͣ�����itpdk�Զ�����(�Զ���ί�кţ���ȷ�ϡ��ɽ���������Ϣ�л������ֵ)
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_GGTEntrust_ASync(const char* lpKhh, const char* lpJys, const char* lpGdh, const char* lpZqdm, int nJylb, int64 nWtsl, double dWtjg, int nDdlx, int nDdjyxz, int nDdsxxz, int nDdyxrq, double dZsxj, int nDdlb, const char* sKZLSH, const char* sKFSDM, const char* sStep, int64 nWTPCH, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_GGTEntrustEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 nWtsl, double dWtjg, int nDdlx, int nDdjyxz, int nDdsxxz, int nDdyxrq, double dZsxj, int nDdlb, const char* sKZLSH, const char* sStep, int64 nWTPCH, int64 nKFSBDBH = 0);

/*
    ����������ί��
        ע����׺Ex�ӿ�ͨ��cusreqinfo��������ͻ��š��ɶ��š��ڵ�š�token
    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ֤ȯ����
    nJylb       [in]        �������(JYLB_BUY��JYLB_SALE)
    lWtsl       [in]        ί������
    lfWtjg      [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx       [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh       [in]        �ɶ���
    return                �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_BJSEntrust(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_BJSEntrustEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx);

/*
    �첽����������ί��
        ע����׺Ex�ӿ�ͨ��cusreqinfo��������ͻ��š��ɶ��š��ڵ�š�token
    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ֤ȯ����
    nJylb       [in]        �������(JYLB_BUY��JYLB_SALE)
    lWtsl       [in]        ί������
    lfWtjg      [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx       [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh       [in]        �ɶ���
    nKFSBDBH    [in]        �����̱��ر�ţ���δ�ͣ�����itpdk�Զ�����(�Զ���ί�кţ���ȷ�ϡ��ɽ���������Ϣ�л������ֵ)
    return              �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_BJSEntrust_ASync(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char* lpGdh, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_BJSEntrustEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, int64 nKFSBDBH = 0);
/*
    ����ͳһ�ӿ�
        ע����׺Ex�ӿ�ͨ��cusreqinfo��������ͻ��š��ɶ��š��ڵ�š�token
    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ֤ȯ����
    nJylb       [in]        �������
    lWtsl       [in]        ί������
    lfWtjg      [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx       [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh       [in]        �ɶ���
    lpStep      [in]        ��չ�ֶ�(Ĭ�Ͽգ�ETF����ʱʹ��)
    nFhfs       [in]        �ֺ췽ʽ(���÷ֺ췽ʽʱ�����ͣ�0 תͶ,1 �ֽ𣬻���ҵ��ʱʹ�ã�
    lpDfjjdm    [in]        Ŀ��������(����ҵ��ʱʹ��)
    lpYSYYSGR   [in]        Ԥ��ҪԼ�չ���(�ǽ���ҵ��ʱʹ��)
    return                �ɹ�����ί�кţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_InterfaceOrder(const char *lpKhh, const char *lpJys, const char *lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char *lpGdh, const char*lpStep = "", int nFhfs = 0, const char* lpDfjjdm = "", const char* lpYSYYSGR = "");
ITPDK_API int64 SECITPDK_InterfaceOrderEx(ITPDK_CusReqInfo& cusreqinfo, const char *lpJys, const char *lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char*lpStep = "", int nFhfs = 0, const char* lpDfjjdm = "", const char* lpYSYYSGR = "");

/*
    �첽����ͳһ�ӿ�
        ע����׺Ex�ӿ�ͨ��cusreqinfo��������ͻ��š��ɶ��š��ڵ�š�token
    lpKhh       [in]        �ͻ���
    lpJys       [in]        ������
    lpZqdm      [in]        ֤ȯ����
    nJylb       [in]        �������
    lWtsl       [in]        ί������
    lfWtjg      [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx       [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh       [in]        �ɶ���
    lpStep      [in]        ��չ�ֶ�(Ĭ�Ͽգ�ETF����ʱʹ��)
    nFhfs       [in]        �ֺ췽ʽ(���÷ֺ췽ʽʱ�����ͣ�0 תͶ,1 �ֽ𣬻���ҵ��ʱʹ�ã�
    lpDfjjdm    [in]        Ŀ��������(����ҵ��ʱʹ��)
    lpYSYYSGR   [in]        Ԥ��ҪԼ�չ���(�ǽ���ҵ��ʱʹ��)
    nKFSBDBH    [in]        �����̱��ر�ţ���δ�ͣ�����itpdk�Զ�����(�Զ���ί�кţ���ȷ�ϡ��ɽ���������Ϣ�л������ֵ)
    return              �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_InterfaceOrder_ASync(const char *lpKhh, const char *lpJys, const char *lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char *lpGdh, const char*lpStep = "", int nFhfs = 0, const char* lpDfjjdm = "", const char* lpYSYYSGR = "", int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_InterfaceOrderEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, int64 lWtsl, double lfWtjg, int64 lDdlx, const char*lpStep = "", int nFhfs = 0, const char* lpDfjjdm = "", const char* lpYSYYSGR = "", int64 nKFSBDBH = 0);

/*
    ��ͨί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_OrderWithdraw(const char* lpKhh, const char* lpJys, int64 nCxwth);
ITPDK_API int64 SECITPDK_OrderWithdrawEx(ITPDK_CusReqInfo &cusreqinfo, const char* lpJys, int64 nCxwth);

/*
    �첽��ͨί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                                ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                ʧ�ܷ���<0���ɹ�����KFSBDBH
*/
ITPDK_API int64 SECITPDK_OrderWithdraw_ASync(const char* lpKhh, const char* lpJys, int64 lCXWth, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_OrderWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 lCXWth, int64 nKFSBDBH = 0);

/*
    ��ͨί�и��ݿ����̱��ر�ų���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxkfsbdh    [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_OrderWithdrawByKFSBDBH(const char* lpKhh, const char* lpJys, int64 nCXKFSBDBH);
ITPDK_API int64 SECITPDK_OrderWithdrawByKFSBDBHEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCXKFSBDBH);

/*
    �첽��ͨί�и��ݿ����̱��ر�ų���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxkfsbdh    [in]        ���������̱��ر��
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                                ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_OrderWithdrawByKFSBDBH_ASync(const char* lpKhh, const char* lpJys, int64 nCXKFSBDBH, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_OrderWithdrawByKFSBDBHEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCXKFSBDBH, int64 nKFSBDBH = 0);

/*
    ��ͨί����������

    lpKhh        [in]        �ͻ���
    nLSH         [in]        ��ˮ�ţ�����ί��ʱ�������ˮ�ţ�Ϊ0��ȫ��ί��
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_BatchOrderWithdraw(const char* lpKhh, int64 nLSH);
ITPDK_API int64 SECITPDK_BatchOrderWithdrawEx(ITPDK_CusReqInfo& cusreqinfo, int64 nLSH);

/*
    �첽��ͨί����������

    lpKhh        [in]        �ͻ���
    nLSH         [in]        ��ˮ�ţ�����ί��ʱ�������ˮ�ţ�Ϊ0��ȫ��ί�У��ڻص��������ͻء�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_BatchOrderWithdraw_ASync(const char* lpKhh, int64 nLSH);
ITPDK_API int64 SECITPDK_BatchOrderWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, int64 nLSH);

/*
    ETFί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lCxwth       [in]        ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ETFWithdraw(const char* lpKhh, const char* lpJys, int64 lCxwth);
ITPDK_API int64 SECITPDK_ETFWithdrawEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 lCxwth);

/*
    �첽ETFί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lWth         [in]        ί�к�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                                ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ETFWithdraw_ASync(const char* lpKhh, const char* lpJys, int64 lCxwth, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_ETFWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 lCxwth, int64 nKFSBDBH = 0);

/*
    ETFί�и��ݿ����̱��ر�ų���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCXKFSBDBH   [in]        �����̱��ر��루���첽ί�нӿڷ��أ�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ETFWithdrawByKFSBDBH(const char* lpKhh, const char* lpJys, int64 nCXKFSBDBH);
ITPDK_API int64 SECITPDK_ETFWithdrawByKFSBDBHEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCXKFSBDBH);

/*
    �첽ETFί�и��ݿ����̱��ر�ų���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCXKFSBDBH   [in]        �����̱��ر��루���첽ί�нӿڷ��أ�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                                ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ETFWithdrawByKFSBDBH_ASync(const char* lpKhh, const char* lpJys, int64 nCXKFSBDBH, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_ETFWithdrawByKFSBDBHEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCXKFSBDBH, int64 nKFSBDBH = 0);

/*
    ծȯ�ع�ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ZQHGWithdraw(const char* lpKhh, const char* lpJys, int64 nCxwth);
ITPDK_API int64 SECITPDK_ZQHGWithdrawEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth);

/*
    �첽ծȯ�ع�ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                            ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ZQHGWithdraw_ASync(const char* lpKhh, const char* lpJys, int64 nCxwth, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_ZQHGWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth, int64 nKFSBDBH = 0);

/*
    ծȯ�ع�ί�и��ݿ����̱��ر�ų���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxkfsbdh    [in]        ���������̱��ر��
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ZQHGWithdrawByKFSBDBH(const char* lpKhh, const char* lpJys, int64 nCxkfsbdh);
ITPDK_API int64 SECITPDK_ZQHGWithdrawByKFSBDBHEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxkfsbdh);

/*
    �첽ծȯ�ع�ί�и��ݿ����̱��ر�ų���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxkfsbdh    [in]        ���������̱��ر��
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                            ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ZQHGWithdrawByKFSBDBH_ASync(const char* lpKhh, const char* lpJys, int64 nCxkfsbdh, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_ZQHGWithdrawByKFSBDBHEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxkfsbdh, int64 nKFSBDBH = 0);

/*
    ����ҵ��ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FXYWWithdraw(const char* lpKhh, const char* lpJys, int64 nCxwth);
ITPDK_API int64 SECITPDK_FXYWWithdrawEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth);

/*
    �첽����ҵ��ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                            ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FXYWWithdraw_ASync(const char* lpKhh, const char* lpJys, int64 nCxwth, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_FXYWWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth, int64 nKFSBDBH = 0);

/*
    LOFҵ�񳷵�

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_LOFWithdraw(const char* lpKhh, const char* lpJys, int64 nCxwth);
ITPDK_API int64 SECITPDK_LOFWithdrawEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth);

/*
    �첽LOFҵ�񳷵�

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                            ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_LOFWithdraw_ASync(const char* lpKhh, const char* lpJys, int64 nCxwth, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_LOFWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth, int64 nKFSBDBH = 0);

/*
    LOFҵ��ί�и��ݿ����̱��ر�ų���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxkfsbdh    [in]        ���������̱��ر��
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_LOFWithdrawByKFSBDBH(const char* lpKhh, const char* lpJys, int64 nCxkfsbdh);
ITPDK_API int64 SECITPDK_LOFWithdrawByKFSBDBHEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxkfsbdh);

/*
    �첽LOFҵ��ί�и��ݿ����̱��ر�ų���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxkfsbdh    [in]        ���������̱��ر��
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                            ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_LOFWithdrawByKFSBDBH_ASync(const char* lpKhh, const char* lpJys, int64 nCxkfsbdh, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_LOFWithdrawByKFSBDBHEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxkfsbdh, int64 nKFSBDBH = 0);

/*
    �̺󶨼�ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_PHDJWithdraw(const char* lpKhh, const char* lpJys, int64 nCxwth);
ITPDK_API int64 SECITPDK_PHDJWithdrawEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth);

/*
    �첽�̺󶨼�ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                                ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                ʧ�ܷ���<0���ɹ����ؿ����̱��ر��
*/
ITPDK_API int64 SECITPDK_PHDJWithdraw_ASync(const char* lpKhh, const char* lpJys, int64 lCXWth, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_PHDJWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 lCXWth, int64 nKFSBDBH = 0);

/*
    �ǽ���ҵ��ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FJYWithdraw(const char* lpKhh, const char* lpJys, int64 nCxwth);
ITPDK_API int64 SECITPDK_FJYWithdrawEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth);

/*
    �첽�ǽ���ҵ��ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                            ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FJYWithdraw_ASync(const char* lpKhh, const char* lpJys, int64 nCxwth, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_FJYWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth, int64 nKFSBDBH = 0);

/*
    �Ϻ�����ͨҵ��ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_SHJJTWithdraw(const char* lpKhh, const char* lpJys, int64 nCxwth);
ITPDK_API int64 SECITPDK_SHJJTWithdrawEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth);

/*
    �첽�Ϻ�����ͨҵ��ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                            ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_SHJJTWithdraw_ASync(const char* lpKhh, const char* lpJys, int64 nCxwth, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_SHJJTWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCxwth, int64 nKFSBDBH = 0);

/*
    �۹�ͨ����

    lpKhh        [in]        �ͻ���
    lpGdh        [in]        �ɶ���
    nCxwth       [in]        ����ί�к�
    nWTPCH       [in]        ί�����κ�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_GGTWithdraw(const char* lpKhh, const char* lpGdh, int64 nCxwth, int64 nWTPCH);
ITPDK_API int64 SECITPDK_GGTWithdrawEx(ITPDK_CusReqInfo& cusreqinfo, int64 nCxwth, int64 nWTPCH);

/*
    �첽�۹�ͨ����

    lpKhh        [in]        �ͻ���
    lpGdh        [in]        �ɶ���
    nCxwth       [in]        ����ί�к�
    nWTPCH       [in]        ί�����κ�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                            ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_GGTWithdraw_ASync(const char* lpKhh, const char* lpGdh, int64 nCxwth, int64 nWTPCH, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_GGTWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, int64 nCxwth, int64 nWTPCH, int64 nKFSBDBH = 0);

/*
    ������ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_BJSWithdraw(const char* lpKhh, const char* lpJys, int64 nCxwth);
ITPDK_API int64 SECITPDK_BJSWithdrawEx(ITPDK_CusReqInfo &cusreqinfo, const char* lpJys, int64 nCxwth);

/*
    �첽������ί�г���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                                ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                ʧ�ܷ���<0���ɹ�����KFSBDBH
*/
ITPDK_API int64 SECITPDK_BJSWithdraw_ASync(const char* lpKhh, const char* lpJys, int64 lCXWth, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_BJSWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 lCXWth, int64 nKFSBDBH = 0);

/*
    ������ί�и��ݿ����̱��ر�ų���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxkfsbdh    [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_BJSWithdrawByKFSBDBH(const char* lpKhh, const char* lpJys, int64 nCXKFSBDBH);
ITPDK_API int64 SECITPDK_BJSWithdrawByKFSBDBHEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCXKFSBDBH);

/*
    �첽������ί�и��ݿ����̱��ر�ų���

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxkfsbdh    [in]        ���������̱��ر��
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                                ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_BJSWithdrawByKFSBDBH_ASync(const char* lpKhh, const char* lpJys, int64 nCXKFSBDBH, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_BJSWithdrawByKFSBDBHEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int64 nCXKFSBDBH, int64 nKFSBDBH = 0);
/*
    ί�г���ͳһ�ӿ�

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_InterfaceOrderWithdraw(const char* lpKhh, const char* lpJys, int nJylb, int64 nCxwth);
ITPDK_API int64 SECITPDK_InterfaceOrderWithdrawEx(ITPDK_CusReqInfo &cusreqinfo, const char* lpJys, int nJylb, int64 nCxwth);

/*
    �첽ί�г���ͳһ�ӿ�

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxwth       [in]        ����ί�к�
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                                ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                ʧ�ܷ���<0���ɹ�����KFSBDBH
*/
ITPDK_API int64 SECITPDK_InterfaceOrderWithdraw_ASync(const char* lpKhh, const char* lpJys, int nJylb, int64 lCXWth, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_InterfaceOrderWithdrawEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int nJylb, int64 lCXWth, int64 nKFSBDBH = 0);

/*
    ���ݿ����̱��ر�ų���ͳһ�ӿ�

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxkfsbdh    [in]        ����ί�к�
    return                �ɹ�����>0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_InterfaceOrderWithdrawByKFSBDBH(const char* lpKhh, const char* lpJys, int nJylb, int64 nCXKFSBDBH);
ITPDK_API int64 SECITPDK_InterfaceOrderWithdrawByKFSBDBHEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int nJylb, int64 nCXKFSBDBH);

/*
    �첽���ݿ����̱��ر�ų���ͳһ�ӿ�

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    nCxkfsbdh    [in]        ���������̱��ر��
    nKFSBDBH     [in]        ��ˮ�ţ�͸���ֶΣ��ڻص�������ͨ����ֵƥ����ã�
                                ��δ�͸��ֶΣ��ɽӿ��ڲ����ɡ�
    return                �ɹ����ؿ����̱��ر�ţ�ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_InterfaceOrderWithdrawByKFSBDBH_ASync(const char* lpKhh, const char* lpJys, int nJylb, int64 nCXKFSBDBH, int64 nKFSBDBH = 0);
ITPDK_API int64 SECITPDK_InterfaceOrderWithdrawByKFSBDBHEx_ASync(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int nJylb, int64 nCXKFSBDBH, int64 nKFSBDBH = 0);

/*
    ��ͨ������ί����������

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(��JYLB_BUY�깺)
    lfWtjg       [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ��깺������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_TradableQty(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_TradableQtyEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx);

/*
    ETF���Ϲ�����

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ETF����
    nJylb        [in]        �������(��JYLB_ETFSG�깺)
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ��깺������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ETFTradableQty(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, const char* lpGdh);
ITPDK_API int64 SECITPDK_ETFTradableQtyEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb);

/*
    ծȯ��ί������

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ����
    nJylb        [in]        �������
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ��깺������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ZQHGTradableQty(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, const char* lpGdh);
ITPDK_API int64 SECITPDK_ZQHGTradableQtyEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb);

/*
    ����ҵ����Ϲ�����

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ETF����
    nJylb        [in]        �������(JYLB_PGJK��ɽɿ�,JYLB_PSSG�����깺,JYLB_BUY��������)
    lfWtjg       [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ��깺������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FXYWTradableQty(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_FXYWTradableQtyEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx);

/*
    LOFҵ���ί������

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ETF����
    nJylb        [in]        �������(JYLB_JJRG�����Ϲ�,JYLB_JJSG�����깺,JYLB_JJSH�������,JYLB_JJFC����ֲ�,JYLB_JJHB����ϲ�)
    lfWtjg       [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ��깺������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_LOFTradableQty(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_LOFTradableQtyEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx);

/*
    �̺󶨼ۿ�ί����������

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(JYLB_PHDJMR��JYLB_PHDJMC)
    lfWtjg       [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ��깺������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_PHDJTradableQty(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_PHDJTradableQtyEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx);

/*
    �ǽ���ҵ�����������

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ETF����
    nJylb        [in]        �������(JYLB_ZZZGתծת�ɡ�JYLB_ZZHSתծ���ۡ�JYLB_YSYYԤ��ҪԼ)
    lfWtjg       [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ��깺������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FJYTradableQty(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_FJYTradableQtyEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx);

/*
    �Ϻ�����ͨҵ�����������

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ETF����
    nJylb        [in]        �������(JYLB_JJRG�����Ϲ���JYLB_JJSG�����깺��JYLB_JJSH������ء�JYLB_JJFHSZ����ֺ����á�JYLB_JJZH����ת����JYLB_ZTG����ת�й�)
    lfWtjg       [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ��깺������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_SHJJTTradableQty(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_SHJJTTradableQtyEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx);

/*
    �۹�ͨ��ί��������ѯ

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpGdh        [in]        �ɶ���
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������
    lDdlx        [in]        ��������
    lfWtjg       [in]        ί�м۸�
    return                �ɹ����ؿ��깺������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_GGTTradableQty(const char* lpKhh, const char* lpJys, const char* lpGdh, const char* lpZqdm, int nJylb, double dWtjg, int nDdlx);
ITPDK_API int64 SECITPDK_GGTTradableQtyEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, double dWtjg, int nDdlx);

/*
    ������������ί����������

    lpKhh        [in]        �ͻ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    nJylb        [in]        �������(��JYLB_BUY�깺)
    lfWtjg       [in]        ί�м۸�(��������Ϊ�м�ʱ���ò�������Ч)
    lDdlx        [in]        ��������(DDLX_XJWT,DDLX_SJWT)
    lpGdh        [in]        �ɶ���
    return                �ɹ����ؿ��깺������ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_BJSTradableQty(const char* lpKhh, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx, const char* lpGdh);
ITPDK_API int64 SECITPDK_BJSTradableQtyEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, int nJylb, double lfWtjg, int32 lDdlx);
/*
    ���������뼤�ע��
    lpKhh        [in]        �ͻ���
    lpGdh        [in]        �ɶ���
    lpJys        [in]        ��������SZ��SH��SB��HB��
    nJylb        [in]        �������JYLB_JHMM�������롢JYLB_ZXMMע�����룩
    nJhm         [in]        ������
    return              �ɹ����ش���0��ʧ�ܷ���<=0
*/
ITPDK_API int64 SECITPDK_PasswordService(const char* lpKhh, const char* lpGdh, const char* lpJys, int nJylb, int64 nJhm);
ITPDK_API int64 SECITPDK_PasswordServiceEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, int nJylb, int64 nJhm);

/*
    ����ͶƱ
    lpKhh        [in]        �ͻ���
    lpGdh        [in]        �ɶ���
    lpJys        [in]        ������
    lpZqdm       [in]        ֤ȯ����
    lpGgbh       [in]        ������
    lpTpya       [in]        ͶƱ�鰸
    nTpsl        [in]        ͶƱ����(��ͨͶƱ������1��2��3��ʾ�޳ɡ����ԡ���Ȩ���ۻ���������)
*/
ITPDK_API int64 SECITPDK_Vote(const char* lpKhh, const char* lpGdh, const char* lpJys, const char* lpZqdm, const char* lpGgbh, const char* lpTpya, int64 nTpsl);
ITPDK_API int64 SECITPDK_VoteEx(ITPDK_CusReqInfo& cusreqinfo, const char* lpJys, const char* lpZqdm, const char* lpGgbh, const char* lpTpya, int64 nTpsl);

/*
    �����޸�

    lpKhh        [in]        �ͻ���
    oldpwd       [in]        ������
    newpwd       [in]        ������
    nType        [in]        �޸����ͣ�0 �޸Ľ������� 1 �޸��ʽ����룩
    return                �ɹ�����true��ʧ�ܷ���false
*/
ITPDK_API bool SECITPDK_ModifyPwd(const char* lpKhh, const char* oldpwd, const char* newpwd, int nType);

/*
    ��ѯ��̨�����ʽ�

    lpKhh        [in]        �ͻ���
    sJymm        [in]        ��������
    sZjzh        [in]        �ʽ��˺�
    return                �ɹ����ؿ����ʽ�ʧ�ܷ���<0
*/
ITPDK_API double SECITPDK_QueryGTFund(const char* sKhh, const char* sJymm, const char* sZjzh = nullptr);

/*
    �ʽ�ӹ�̨����

    lpKhh        [in]        �ͻ���
    sJymm        [in]        ��������
    dFsje        [in]        �������
    sZjzh        [in]        �ʽ��˺�
    sGyh         [in]        ��Ա��
    sGymm        [in]        ��Ա����
    sZy          [in]        ժҪ
    return                �ɹ�����>=0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FundTransIn(const char* sKhh, const char* sJymm, double dFsje, const char* sZjzh, const char* sGyh = nullptr, const char* sGymm = nullptr, const char* sZy = nullptr);

/*
    �ʽ��������̨

    lpKhh        [in]        �ͻ���
    sJymm        [in]        ��������
    dFsje        [in]        �������
    sZjzh        [in]        �ʽ��˺�
    sGyh         [in]        ��Ա��
    sGymm        [in]        ��Ա����
    sZy          [in]        ժҪ
    return                �ɹ�����>=0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FundTransOut(const char* sKhh, const char* sJymm, double dFsje, const char* sZjzh, const char* sGyh = nullptr, const char* sGymm = nullptr, const char* sZy = nullptr);

/*
    ֤ȯ�ӹ�̨����

    lpKhh        [in]        �ͻ���
    sJys         [in]        ������
    sJymm        [in]        ��������
    nFssl        [in]        ��������
    sZqdm        [in]        ֤ȯ����
    sGdh         [in]        �ɶ���
    sZy          [in]        ժҪ
    return                �ɹ�����>=0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_StockTransIn(const char* sKhh, const char* sJymm, const char* sJys, double nFssl, const char* sZqdm, const char* sGdh, const char* sZy = nullptr);

/*
    ֤ȯ��������̨

    lpKhh        [in]        �ͻ���
    sJys         [in]        ������
    sJymm        [in]        ��������
    nFssl        [in]        ��������
    sZqdm        [in]        ֤ȯ����
    sGyh         [in]        �ɶ���
    sZy          [in]        ժҪ
    return                �ɹ�����>=0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_StockTransOut(const char* sKhh, const char* sJymm, const char* sJys, int nFssl, const char* sZqdm, const char* sGdh, const char* sZy = nullptr);

/*
    ��̨�ʽ������ϸ

    lpKhh        [in]        �ͻ���
    sJymm        [in]        ��������
    sZjzh        [in]        �ʽ��˺�
    return                �ɹ�����>=0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FundTransDetail(const char* sKhh, const char* sJymm, const char* sZjzh, int nRowcount, const char* lpBrowindex, vector<ITPDK_GTZJLS>& arGTZJLS);

/*
    һ������ - �ڵ���ʽ𻮲�

    lpKhh        [in]        �ͻ���
    lpJymm       [in]        ��������
    sZjzh        [in]        �ʽ��˺�
    dFsje        [in]        �������
    nNode1       [in]        �����ڵ�
    nNode2       [in]        ����ڵ�
    sZy          [in]        ժҪ
    return                �ɹ�����>=0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_FundTransBetweenNodes(const char* lpKhh, const char* sJymm, const char* sZjzh, double dFsje, int64 nNode1, int64 nNode2, const char* sZy);

/*
    һ������ - �������Բ�ѯ

    lpKhh        [in]        �ͻ���
    lpJymm       [in]        ��������
    sZjzh        [in]        �ʽ��˺�
    arZJHBCL     [out]        �ʽ𻮲�������Ϣ
    return                �ɹ�����>=0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_QueryZJHBCL(const char* lpKhh, const char* sJymm, const char* sZjzh, vector<ITPDK_ZJHBCL>& arZJHBCL);

/*
    һ������ - �޸��ʽ����

    lpKhh        [in]        �ͻ���
    lpJymm       [in]        ��������
    lpZjzh       [in]        �ʽ��˺�
    nNode1       [in]        �ڵ�1���
    nNode2       [in]        �ڵ�2���
    dBL1         [in]        �ڵ�1�ʽ����
    dBL2         [in]        �ڵ�2�ʽ����
    sZy          [in]        ժҪ
    return                �ɹ�����>=0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_ModifyZJHBCL(const char* lpKhh, const char* sJymm, const char* lpZjzh, int64 nNode1, int64 nNode2, double dBL1, double dBL2, const char* sZy);

/*
    һ������ - �ڵ�ɻ����ʽ��Ȳ�ѯ

    lpKhh        [in]        �ͻ���
    lpJymm       [in]        ��������
    lpZjzh       [in]        �ʽ��˺�
    nNode        [in]        �ڵ���
    dKYZJ        [in]        �����ʽ�
    return                �ɹ�����>=0��ʧ�ܷ���<0
*/
ITPDK_API double SECITPDK_QueryJDKYZJ(const char* lpKhh, const char* sJymm, const char* lpZjzh, int64 nNode, double dKYZJ);

/*
    һ������ - �����ʽ𻮲�����

    lpKhh        [in]        �ͻ���
    lpJymm       [in]        ��������
    lpZjzh       [in]        �ʽ��˺�
    nNode        [in]        �ڵ���
    dBL          [in]        �ʽ����
    sZy          [in]        ժҪ
    return                �ɹ�����>=0��ʧ�ܷ���<0
*/
ITPDK_API int64 SECITPDK_AddZJHBCL(const char* lpKhh, const char* lpJymm, const char* lpZjzh, int64 nNode, double dBL, const char* sZy);

/*
    һ������ - �ڵ���ʽ𻮲���ϸ��ѯ

    lpKhh        [in]        �ͻ���
    lpJymm       [in]        ��������
    lpZjzh       [in]        �ʽ��˺�
    nRowcount    [in]        ���ؼ�¼����0ʱ����ȫ����
    nBrowindex   [in]        ��ҳ��0ʱ��ͷ�飩
    arZJHBMX     [out]        ��ѯ���
    return                ���ز�ѯ���ļ�¼��
*/
ITPDK_API int64 SECITPDK_QueryJDJZJHBMX(const char* lpKhh, const char* lpJymm, const char* lpZjzh, int nRowcount, const char* lpBrowindex, vector<ITPDK_JDJZJHBMX>& arZJHBMX);

