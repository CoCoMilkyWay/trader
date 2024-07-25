//----------------------------------------------------------------------------
// ��Ȩ������������ģ�����ڽ�֤΢�ں˼ܹ�ƽ̨(KMAP)��һ����
//           ��֤�Ƽ��ɷ����޹�˾  ��Ȩ����
//
// �ļ����ƣ�maCliApi.h
// ģ�����ƣ�ma΢�ܹ��ͻ���(Client)��C����API�ӿ�
// ģ��������
// �������ߣ������
// �������ڣ�2012-10-10
// ģ��汾��001.000.000
//----------------------------------------------------------------------------
// �޸�����      �汾          ����            ��ע
//----------------------------------------------------------------------------
// 2012-10-10    1.0          �����          ԭ��
// 2012-10-17    1.1          ½����          ʵ�־��幦��
// 2013-09-16    1.2          ������          �������幦��
// 2013-12-27    1.3          ������          ���Ӽ��ܺ���
// 2014-11-21    1.4          �����          ���ӿͻ�����־����
// 2015-01-26    1.5          �����          ����Ĭ�ϻص�����������ȫ���ɽ���Ϣ
// 2015-03-28    1.6          �����          ���ӻص���Ӧ�������ļ���/
// 2015-07-17    1.7          �����          ������־ģʽ
// 2016-08-16    2.0          �Ŷ���          ֧��SSLͨ�ż���
// 2019-03-23    3.0          �����          �����ļ����书��
// 2019-08-15    3.1          ���ľ�          ɾ��ͨ������1-KCXP/2-0MQ/4-SHM����
// 2019-08-15    3.2          ���ľ�          ����MACLI_OPTION_LPORT��MACLI_OPTION_IPORTѡ��
// 2022-11-10    3.9          �����          ����MACLI_HEAD_FID_REQTIME��MACLI_HEAD_FID_ANSTIME��MACLI_HEAD_FID_TRACEID��MACLI_HEAD_FID_SPANDIDͨ�Ű�ͷ�ֶ���������·����
//----------------------------------------------------------------------------
#if !defined(__MA_CLI_API_H__)
#define __MA_CLI_API_H__

#if defined(WIN32) || defined(WIN64) || defined(OS_IS_WINDOWS)
  #if defined(MA_CLIAPI_EXPORTS)
    #define MACLIAPI __declspec(dllexport)
  #else
    #define MACLIAPI __declspec(dllimport)
  #endif
  #define MACLI_STDCALL __stdcall
  #define MACLI_EXPORTS __declspec(dllexport)
#else
  #define MACLIAPI
  #define MACLI_STDCALL
  #define MACLI_EXPORTS
  #if !defined __int64
    #define __int64 long long
  #endif
#endif

//////////////////////////////////////////////////////////////////////////
//ͨ�Ű�ͷ�ֶ�����
#define MACLI_HEAD_FID_PKT_LEN              0         //(ֻ��)  p_pvdValuePtr=(int *)                     p_iValueSize=sizeof(int)
#define MACLI_HEAD_FID_PKT_CRC              1         //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >4
#define MACLI_HEAD_FID_PKT_ID               2         //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >4
#define MACLI_HEAD_FID_PKT_VER              3         //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >2
#define MACLI_HEAD_FID_PKT_TYPE             4         //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >1
#define MACLI_HEAD_FID_MSG_TYPE             5         //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >1
#define MACLI_HEAD_FID_RESEND_FLAG          6         //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >1
#define MACLI_HEAD_FID_TIMESTAMP            7         //(ֻ��)  p_pvdValuePtr=(void *)(__int64 *)         p_iValueSize=sizeof(__int64)
#define MACLI_HEAD_FID_MSG_ID               8         //(��д)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >32 [д] strlen(p_pvdValuePtr)
#define MACLI_HEAD_FID_CORR_ID              9         //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >32
//#define MACLI_HEAD_FID_MSG_INDEX          10        //(ֻ��)
#define MACLI_HEAD_FID_FUNC_ID              11        //(��д)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >32 [д] strlen(p_pvdValuePtr)
#define MACLI_HEAD_FID_SRC_NODE             12        //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >4
#define MACLI_HEAD_FID_DEST_NODE            13        //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >4
#define MACLI_HEAD_FID_PAGE_FLAG            14        //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >1
#define MACLI_HEAD_FID_PAGE_NO              15        //(ֻ��)  p_pvdValuePtr=(void *)(int  *)            p_iValueSize=sizeof(int)
#define MACLI_HEAD_FID_PAGE_CNT             16        //(ֻ��)  p_pvdValuePtr=(void *)(int  *)            p_iValueSize=sizeof(int)
#define MACLI_HEAD_FID_BODY_LEN             21        //(ֻ��)  p_pvdValuePtr=(void *)(int  *)            p_iValueSize=sizeof(int)
#define MACLI_HEAD_FID_PKT_HEAD_END         25        //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >4
#define MACLI_HEAD_FID_PKT_HEAD_LEN         35        //(ֻ��)  p_pvdValuePtr=(void *)(int  *)            p_iValueSize=sizeof(int)
#define MACLI_HEAD_FID_PKT_HEAD_MSG         41        //(ֻ��)  p_pvdValuePtr=(void *)(char  *)           p_iValueSize=[��] > ͨ��MACLI_HEAD_FID_PKT_HEAD_END��ȡ�õ�����ֵ
#define MACLI_HEAD_FID_PKT_BODY_MSG         42        //(ֻ��)  p_pvdValuePtr=(void *)(char  *)           p_iValueSize=[��] > ͨ��MACLI_HEAD_FID_PKT_LEN��ȡ�õ�����ֵ - ͨ��MACLI_HEAD_FID_PKT_HEAD_END��ȡ�õ�����ֵ
#define MACLI_HEAD_FID_PKT_MSG              43        //(ֻ��)  p_pvdValuePtr=(void *)(char  *)           p_iValueSize=[��] > ͨ��MACLI_HEAD_FID_PKT_LEN��ȡ�õ�����ֵ
#define MACLI_HEAD_FID_REQTIME              48        //(ֻ��)  p_pvdValuePtr=(void *)(__int64 *)         p_iValueSize=sizeof(__int64)
#define MACLI_HEAD_FID_ANSTIME              49        //(ֻ��)  p_pvdValuePtr=(void *)(__int64 *)         p_iValueSize=sizeof(__int64)
#define MACLI_HEAD_FID_TRACEID              50        //(��д)  p_pvdValuePtr=(void *)(__int64 *)         p_iValueSize=sizeof(__int64)
#define MACLI_HEAD_FID_SPANDID              51        //(��д)  p_pvdValuePtr=(void *)(__int64 *)         p_iValueSize=sizeof(__int64)
#define MACLI_HEAD_FID_FUNC_TYPE            1052672   //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >1
#define MACLI_HEAD_FID_BIZ_CHANNEL          1052674   //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >2
#define MACLI_HEAD_FID_TOKEN_FLAG           1069056   //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >1
#define MACLI_HEAD_FID_PUB_TOPIC            1073152   //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >12
#define MACLI_HEAD_FID_PUB_KEY1             1073153   //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >32
#define MACLI_HEAD_FID_PUB_KEY2             1073154   //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >32
#define MACLI_HEAD_FID_PUB_KEY3             1073155   //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >32
#define MACLI_HEAD_FID_USER_SESSION         1871872   //(ֻ��)  p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >64

//////////////////////////////////////////////////////////////////////////
//ѡ������
#define MACLI_OPTION_CONNECT_PARAM          1         //(��д) p_pvdValuePtr=(ST_MACLI_CONNECT_OPTION *) p_iValueSize=sizeof(ST_MACLI_CONNECT_OPTION)
#define MACLI_OPTION_SYNCCALL_TIMEOUT       2         //(��д) p_pvdValuePtr=(int *)                     p_iValueSize=sizeof(int)
#define MACLI_OPTION_ASYNCALL_TIMEOUT       3         //(��д) p_pvdValuePtr=(int *)                     p_iValueSize=sizeof(int)
#define MACLI_OPTION_WRITELOG_LEVEL         4         //(��д) p_pvdValuePtr=(int *)                     p_iValueSize=sizeof(int)
#define MACLI_OPTION_WRITELOG_PATH          5         //(��д) p_pvdValuePtr=(void *)(char *)            p_iValueSize=[��] >256 [д] strlen(p_pvdValuePtr)
#define MACLI_OPTION_WRITELOG_SIZE          6         //(��д) p_pvdValuePtr=(int *)                     p_iValueSize=sizeof(int) [��λMB]
#define MACLI_OPTION_WRITELOG_TIME          7         //(��д) p_pvdValuePtr=(int *)                     p_iValueSize=sizeof(int) [��]
#define MACLI_OPTION_WRITELOG_MODE          8         //(��д) p_pvdValuePtr=(int *)                     p_iValueSize=sizeof(int)
#define MACLI_OPTION_SSL_CONNECTION         9         //(��д) p_pvdValuePtr=(int *)                     p_iValueSize=sizeof(int) [�Ƿ�֧��SSL����ͨ��,1����0Ϊ�Ǽ���]
#define MACLI_OPTION_LIP_ADDR               10        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_MAC_ADDR               11        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_APP_NAME               12        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_HD_ID                  13        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_CPU_INFO               14        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_CPU_ID                 15        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_PC_NAME                16        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_HD_PART                17        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_SYS_VOL                18        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_OS_VER                 19        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_SIP_ADDR               20        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_LPORT                  21        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256
#define MACLI_OPTION_IPORT                  22        //(��)  p_pvdValuePtr=(char *)                     p_iValueSize=[��] >256

//////////////////////////////////////////////////////////////////////////
// ��־ģʽ����:Ĭ��Ϊѭ����־
#define MACLI_WRITELOG_MODE_LOOP            0         //ѭ��ģʽ����־�ļ��ﵽָ����С���������д����־�ļ�)[maCliApi.log]
#define MACLI_WRITELOG_MODE_FORWARD         1         //����ģʽ����־�ļ��ﵽָ����С����дһ����־�ļ���   [maCliApixxxx.log] xxxx��0000��9999����

//////////////////////////////////////////////////////////////////////////
// ��־�����壨���õ��ӷ�ʽ��:���� ������Ϣ[1] + ������Ϣ[8] = 9
#define MACLI_WRITELOG_LEVEL_NOLOG          0         //����־
#define MACLI_WRITELOG_LEVEL_ALLLOG         63        //������־
#define MACLI_WRITELOG_LEVEL_DEBUG          1         //������Ϣ(�շ���������������)
#define MACLI_WRITELOG_LEVEL_INFO           2         //������Ϣ(�շ�����Ҫ���ݣ����ܺš�����С)
#define MACLI_WRITELOG_LEVEL_WARN           4         //������Ϣ(�շ���ͨ�ŶϿ��������������л�����Ϣ��ʱ��)
#define MACLI_WRITELOG_LEVEL_ERROR          8         //������Ϣ(�շ���ͨ��ʧ��)
#define MACLI_WRITELOG_LEVEL_FATAL          16        //���ش���(�����쳣�˳�)
#define MACLI_WRITELOG_LEVEL_IMPORTANT      32        //��Ҫ��Ϣ(���ӽ��������ӹرյ�)


//////////////////////////////////////////////////////////////////////////
// ͨ�����Ͷ���
//#define COMM_TYPE_KCXP                    1       // ����(maCliApi 3.1)
//#define COMM_TYPE_ZMQ                     2       // ����(maCliApi 3.1)
#define COMM_TYPE_SOCKET                    3
//#define COMM_TYPE_SHM                     4       // ����(maCliApi 3.1)

//////////////////////////////////////////////////////////////////////////
// ���ù��ܶ���
#define MA_FUNC_SUBSCRIBE                   "00102012"  //����
#define MA_FUNC_UNSUBSCRIBE                 "00102013"  //ȡ������
#define MA_FUNC_PUB_CONTENT                 "00102020"  //����
#define MA_FUNC_LOGIN_API                   "10301105"  //�û���¼(API)
#define MA_FUNC_LOGIN_ACCREDIT              "10301104"  //�ͻ����ŵ�¼

//////////////////////////////////////////////////////////////////////////
// ��������״̬����
#define MA_NET_CONNET                       "0"       //���ӳɹ�
#define MA_NET_BREAK                        "1"       //���ӶϿ�

//////////////////////////////////////////////////////////////////////////
//�ص�����
typedef void *MACLIHANDLE;
// �������ݺ���ԭ����:OnPulish(const char *p_pszAcceptSn, const unsigned char *p_pszDataBuff, int p_iDataLen) 
// �첽Ӧ����ԭ����:OnAnswer(const char *p_pszMsgId, const unsigned char *p_pszDataBuff, int p_iDataLen) 
// �ļ��ϴ�����ԭ����:OnFileUp(const char *p_pszFileName, const unsigned char *p_pszError, int p_iPercent)   // p_iPercent > 0 ��������  p_iPercent < 0 �쳣
// �ļ����غ���ԭ����:OnFileDown(const char *p_pszFileName, const unsigned char *p_pszError, int p_iPercent) // p_iPercent > 0 ��������  p_iPercent < 0 �쳣
typedef void (MACLI_Callback) (const char *, const unsigned char *, int);
typedef MACLI_Callback *MACLI_NOTIFY;

#define MACLI_SERVERNAME_MAX                32
#define MACLI_ADRRESS_MAX                   128
#define MACLI_PROXY_MAX                     128
#define MACLI_SSL_MAX                       256
#define MACLI_MSG_ID_SIZE                   32
#define MACLI_TOPIC_MAX                     12
#define MACLI_TERM_CODE                     32
#define MACLI_USERDATA_MAX                  256
#define MACLI_USERINFO_MAX                  64
#define MACLI_FUNC_ID_SIZE                  8   //���ܺ�
#define MACLI_PUB_TOPIC_SIZE                12  //����

#if (defined(OS_IS_AIX) && defined(__xlC__))
#pragma options align = packed
#else
#pragma pack(1)
#endif

typedef struct
{
  char szServerName[MACLI_SERVERNAME_MAX + 1];  //����������
  int nCommType;                                //ͨ�����ͣ�1-KCXP 2-0MQ 3-SOCKET 4-SHM
  int nProtocal;                                //ͨ��Э�飺1-TCP 2-UDP 3-EPGM 4-IPC
  char szSvrAddress[MACLI_ADRRESS_MAX + 1];     //��������ַ
  int nSvrPort;                                 //�������˿�
  char szLocalAddress[MACLI_ADRRESS_MAX + 1];   //���ص�ַ
  int nLocalPort;                               //���ض˿�

  int nReqId;                                   //�����ʶ��ͨ������Ϊ4-SHMʱΪ������б��
  int nAnsId;                                   //Ӧ���ʶ��ͨ������Ϊ4-SHMʱΪӦ����б��

  char szSubId[MACLI_SERVERNAME_MAX + 1];       //���ı�ʶ: ͨ������ΪKCXPʱΪ���Ķ���

  char szProxy[MACLI_PROXY_MAX + 1];            //���������(����)
  char szSSL[MACLI_SSL_MAX + 1];                //SSL(����)
  char szTermCode[MACLI_TERM_CODE + 1];         //�ն�������

  char szReqName[MACLI_SERVERNAME_MAX + 1];     //ͨ������Ϊ4-SHMʱ��д�����ڴ������������
  char szAnsName[MACLI_SERVERNAME_MAX + 1];     //ͨ������Ϊ4-SHMʱ��д�����ڴ�Ӧ���������
  char szReqConnstr[MACLI_ADRRESS_MAX + 1];     //ͨ������Ϊ4-SHMʱ��д�����ڴ�����������Ӵ�, ����IPC/@@IP
  char szAnsConnstr[MACLI_ADRRESS_MAX + 1];     //ͨ������Ϊ4-SHMʱ��д�����ڴ�Ӧ��������Ӵ�

  char szMonitorReqName[MACLI_SERVERNAME_MAX + 1];     //���������������
  char szMonitorAnsName[MACLI_SERVERNAME_MAX + 1];     //����Ӧ���������
  char szMonitorReqConnstr[MACLI_ADRRESS_MAX + 1];     //��������������Ӵ�, ����IPC/@@IP
  char szMonitorAnsConnstr[MACLI_ADRRESS_MAX + 1];     //����Ӧ��������Ӵ�

  char szSocketSubConnstr[MACLI_ADRRESS_MAX + 1];      //SOCKET���Ķ������Ӵ�

  int nReqMaxDepth;                             //ͨ������Ϊ4-SHMʱ��д�����ڴ��������������
  int nAnsMaxDepth;                             //ͨ������Ϊ4-SHMʱ��д�����ڴ�Ӧ�����������
} ST_MACLI_CONNECT_OPTION;

typedef struct
{
  char szFuncId[MACLI_FUNC_ID_SIZE + 1];        //���ܺ�
  MACLI_NOTIFY pfnCallback;                     //�첽Ӧ��ص�����ָ��
} ST_MACLI_ARCALLBACK;

typedef struct
{
  char szTopic[MACLI_PUB_TOPIC_SIZE + 1];       //����
  MACLI_NOTIFY pfnCallback;                     //�������ݻص�����ָ��
} ST_MACLI_PSCALLBACK;

typedef struct
{
  MACLI_NOTIFY pfnCallback;                     //������Ϣ�ص�����ָ��
} ST_MACLI_NETCALLBACK;

typedef struct
{
  char szServerName[MACLI_USERINFO_MAX + 1];    //��������
  char szUserId    [MACLI_USERINFO_MAX + 1];    //�û���  
  char szPassword  [MACLI_USERINFO_MAX + 1];    //����    
  char szAppId     [MACLI_USERINFO_MAX + 1];    //APP��ʶ 
  char szAuthCode  [MACLI_USERINFO_MAX + 1];    //��֤��  
} ST_MACLI_USERINFO;

typedef struct
{
  char szFuncId[MACLI_FUNC_ID_SIZE + 1];        //���ܺ�
  char szMsgId[MACLI_MSG_ID_SIZE + 1];          //��ϢID
  int nTimeout;                                 //���ó�ʱ
  char szUserData1[MACLI_USERDATA_MAX + 1];     //�û�����1
  char szUserData2[MACLI_USERDATA_MAX + 1];     //�û�����2
} ST_MACLI_SYNCCALL;

typedef struct
{
  char szFuncId[MACLI_FUNC_ID_SIZE + 1];        //���ܺ�
  char szMsgId[MACLI_MSG_ID_SIZE + 1];          //��ϢID
  int nTimeout;                                 //���ó�ʱ
  char szUserData1[MACLI_USERDATA_MAX + 1];     //�û�����1
  char szUserData2[MACLI_USERDATA_MAX + 1];     //�û�����2
} ST_MACLI_ASYNCALL;

typedef struct
{
  char szTopic[MACLI_PUB_TOPIC_SIZE + 1];       //����
  char szAcceptSn[MACLI_MSG_ID_SIZE + 1];       //���������
  int nTimeout;                                 //���ó�ʱ
  char szUserData1[MACLI_USERDATA_MAX + 1];     //�û�����1
  char szUserData2[MACLI_USERDATA_MAX + 1];     //�û�����2
} ST_MACLI_PUBCALL;


#if (defined(OS_IS_AIX) && defined(__xlC__))
#pragma options align = reset
#else
#pragma pack()
#endif

#ifdef __cplusplus
extern "C"
{
#endif

//��ʼ��/����
MACLIAPI int MACLI_STDCALL maCli_Init(MACLIHANDLE *p_phHandle);
MACLIAPI int MACLI_STDCALL maCli_Exit(MACLIHANDLE p_hHandle);

//�汾/����
MACLIAPI int MACLI_STDCALL maCli_GetVersion(MACLIHANDLE p_hHandle, char *p_pszVer, int p_iVerSize);
MACLIAPI int MACLI_STDCALL maCli_GetOptions(MACLIHANDLE p_hHandle, int p_iOptionIdx, void *p_pvdValuePtr, int p_iValueSize);
MACLIAPI int MACLI_STDCALL maCli_SetOptions(MACLIHANDLE p_hHandle, int p_iOptionIdx, void *p_pvdValuePtr, int p_iValueSize);
MACLIAPI int MACLI_STDCALL maCli_SetArCallback(MACLIHANDLE p_hHandle, ST_MACLI_ARCALLBACK *p_pstArCallback);
MACLIAPI int MACLI_STDCALL maCli_SetPsCallback(MACLIHANDLE p_hHandle, ST_MACLI_PSCALLBACK *p_pstPsCallback);
MACLIAPI int MACLI_STDCALL maCli_SetNetCallback(MACLIHANDLE p_hHandle, ST_MACLI_NETCALLBACK *p_pstNetCallback);

//����
MACLIAPI int MACLI_STDCALL maCli_Open(MACLIHANDLE p_hHandle, ST_MACLI_USERINFO *p_pstUserInfo);
MACLIAPI int MACLI_STDCALL maCli_MonitorOpen(MACLIHANDLE p_hHandle, ST_MACLI_USERINFO *p_pstUserInfo);
MACLIAPI int MACLI_STDCALL maCli_Close(MACLIHANDLE p_hHandle);
MACLIAPI int MACLI_STDCALL maCli_ReOpen(MACLIHANDLE p_hHandle);

//����1
MACLIAPI int MACLI_STDCALL maCli_SyncCall(MACLIHANDLE p_hHandle, ST_MACLI_SYNCCALL *p_pstSyncCall);
MACLIAPI int MACLI_STDCALL maCli_AsynCall(MACLIHANDLE p_hHandle, ST_MACLI_ASYNCALL *p_pstAsynCall);
MACLIAPI int MACLI_STDCALL maCli_AsynGetReply(MACLIHANDLE p_hHandle, ST_MACLI_ASYNCALL *p_pstAsynCall);
MACLIAPI int MACLI_STDCALL maCli_GetPsContent(MACLIHANDLE p_hHandle, ST_MACLI_PUBCALL *p_pstPubCall);

//����2
MACLIAPI int MACLI_STDCALL maCli_SyncCall2(MACLIHANDLE p_hHandle, ST_MACLI_SYNCCALL *p_pstSyncCall,
  const unsigned char *p_pszReqData, int p_iReqDataLen, unsigned char **p_ppszAnsData, int *p_piAnsDataLen);
MACLIAPI int MACLI_STDCALL maCli_AsynCall2(MACLIHANDLE p_hHandle, ST_MACLI_ASYNCALL *p_pstAsynCall,
  const unsigned char *p_pszReqData, int p_iReqDataLen);
MACLIAPI int MACLI_STDCALL maCli_AsynGetReply2(MACLIHANDLE p_hHandle, ST_MACLI_ASYNCALL *p_pstAsynCall,
  unsigned char **p_ppszAnsData, int *p_piAnsDataLen);
MACLIAPI int MACLI_STDCALL maCli_GetPsContent2(MACLIHANDLE p_hHandle, ST_MACLI_PUBCALL *p_pstPubCall,
  unsigned char **p_ppszPubData, int *p_piPubDataLen);

MACLIAPI int MACLI_STDCALL maCli_AsynMonitorCall2(MACLIHANDLE p_hHandle, ST_MACLI_ASYNCALL *p_pstAsynCall,
  const unsigned char *p_pszReqData, int p_iReqDataLen);
MACLIAPI int MACLI_STDCALL maCli_AsynMonitorGetReply2(MACLIHANDLE p_hHandle, ST_MACLI_ASYNCALL *p_pstAsynCall,
  unsigned char **p_ppszAnsData, int *p_piAnsDataLen);

//���(����)
MACLIAPI int MACLI_STDCALL maCli_Make(MACLIHANDLE p_hHandle, unsigned char **p_ppszReqData, int *p_piReqDataLen);
MACLIAPI int MACLI_STDCALL maCli_BeginWrite(MACLIHANDLE p_hHandle);
MACLIAPI int MACLI_STDCALL maCli_EndWrite(MACLIHANDLE p_hHandle);
MACLIAPI int MACLI_STDCALL maCli_CreateTable(MACLIHANDLE p_hHandle);
MACLIAPI int MACLI_STDCALL maCli_AddRow(MACLIHANDLE p_hHandle);
MACLIAPI int MACLI_STDCALL maCli_SaveRow(MACLIHANDLE p_hHandle);

MACLIAPI int MACLI_STDCALL maCli_SetHdrValue(MACLIHANDLE p_hHandle, const unsigned char *p_pszValue, int p_iValLen, int p_iFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_SetHdrValueS(MACLIHANDLE p_hHandle, const char *p_pszValue, int p_iFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_SetHdrValueN(MACLIHANDLE p_hHandle, int p_iValue, int p_iFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_SetHdrValueC(MACLIHANDLE p_hHandle, char p_chValue, int p_iFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_SetHdrValueD(MACLIHANDLE p_hHandle, double p_dValue, int p_iFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_SetHdrValueL(MACLIHANDLE p_hHandle, __int64 p_i64Value, int p_iFieldIdx);

MACLIAPI int MACLI_STDCALL maCli_SetValue(MACLIHANDLE p_hHandle, const unsigned char *p_pszValue, int p_iValLen, const char *p_pszFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_SetValueS(MACLIHANDLE p_hHandle, const char *p_pszValue, const char *p_pszFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_SetValueN(MACLIHANDLE p_hHandle, int p_iValue, const char *p_pszFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_SetValueC(MACLIHANDLE p_hHandle, char p_chValue, const char *p_pszFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_SetValueD(MACLIHANDLE p_hHandle, double p_dValue, const char *p_pszFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_SetValueL(MACLIHANDLE p_hHandle, __int64 p_i64Value, const char *p_pszFieldIdx);

//���(Ӧ��)
MACLIAPI int MACLI_STDCALL maCli_Parse(MACLIHANDLE p_hHandle, const unsigned char *p_pszAnsData, int p_iAnsDataLen);
MACLIAPI int MACLI_STDCALL maCli_GetTableCount(MACLIHANDLE p_hHandle, int *p_piTableCount);
MACLIAPI int MACLI_STDCALL maCli_OpenTable(MACLIHANDLE p_hHandle, int p_iTableIndex);
MACLIAPI int MACLI_STDCALL maCli_GetRowCount(MACLIHANDLE p_hHandle, int *p_piRowCount);
MACLIAPI int MACLI_STDCALL maCli_ReadRow(MACLIHANDLE p_hHandle, int p_iRowIndex);

MACLIAPI int MACLI_STDCALL maCli_GetHdrValue(MACLIHANDLE p_hHandle, unsigned char *p_pszValue, int p_iValSize, int *p_piValLen, int p_iFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_GetHdrValueS(MACLIHANDLE p_hHandle, char *p_pszValue, int p_iValSize, int p_iFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_GetHdrValueN(MACLIHANDLE p_hHandle, int *p_piValue, int p_iFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_GetHdrValueC(MACLIHANDLE p_hHandle, char *p_pchValue, int p_iFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_GetHdrValueD(MACLIHANDLE p_hHandle, double *p_pdValue, int p_iFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_GetHdrValueL(MACLIHANDLE p_hHandle, __int64 *p_pi64Value, int p_iFieldIdx);

MACLIAPI int MACLI_STDCALL maCli_GetValue(MACLIHANDLE p_hHandle, unsigned char *p_pszValue, int p_iValSize, int *p_piValLen, const char *p_pszFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_GetValueS(MACLIHANDLE p_hHandle, char *p_pszValue, int p_iValSize, const char *p_pszFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_GetValueN(MACLIHANDLE p_hHandle, int *p_piValue, const char *p_pszFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_GetValueC(MACLIHANDLE p_hHandle, char *p_pchValue, const char *p_pszFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_GetValueD(MACLIHANDLE p_hHandle, double *p_pdValue, const char *p_pszFieldIdx);
MACLIAPI int MACLI_STDCALL maCli_GetValueL(MACLIHANDLE p_hHandle, __int64 *p_pi64Value, const char *p_pszFieldIdx);

//����
MACLIAPI int MACLI_STDCALL maCli_GetUuid(MACLIHANDLE p_hHandle, char *p_pszUuid, int p_iUuidSize);
MACLIAPI int MACLI_STDCALL maCli_RestorePsList(MACLIHANDLE p_hHandle);
MACLIAPI int MACLI_STDCALL maCli_GetLastErrorCode(MACLIHANDLE p_hHandle, int *p_piErrorCode);
MACLIAPI int MACLI_STDCALL maCli_GetLastErrorMsg(MACLIHANDLE p_hHandle, char *p_pszErrorMsg, int p_iMsgSize);
MACLIAPI int MACLI_STDCALL maCli_ComEncrypt(MACLIHANDLE p_hHandle, char *p_pszOutput, int p_iSize, const char *p_pszInput, const char *p_pszKey);
MACLIAPI int MACLI_STDCALL maCli_ComDecrypt(MACLIHANDLE p_hHandle, char *p_pszOutput, int p_iSize, const char *p_pszInput, const char *p_pszKey);
MACLIAPI int MACLI_STDCALL maCli_GetConnIpAddr(MACLIHANDLE p_hHandle, char *p_pszIpAddr, int p_iIpAddrSize);
MACLIAPI int MACLI_STDCALL maCli_GetConnMac(MACLIHANDLE p_hHandle, char *p_pszMac, int p_iMacSize);
MACLIAPI int MACLI_STDCALL maCli_FileUpload(MACLIHANDLE p_hHandle, const char *p_pszFileId, const char *p_pszLocalFileName, MACLI_NOTIFY p_pfnCbProgress);
MACLIAPI int MACLI_STDCALL maCli_FileDownload(MACLIHANDLE p_hHandle, const char *p_pszFileId, const char *p_pszLocalFileName, MACLI_NOTIFY p_pfnCbProgress);
MACLIAPI int MACLI_STDCALL maCli_FolderUpload(MACLIHANDLE p_hHandle, const char *p_pszFileId, const char *p_pszLocalFolder, MACLI_NOTIFY p_pfnCbProgress);
MACLIAPI int MACLI_STDCALL maCli_FolderDownload(MACLIHANDLE p_hHandle, const char *p_pszFileId, const char *p_pszLocalFolder, MACLI_NOTIFY p_pfnCbProgress);

#ifdef __cplusplus
}
#endif

#endif  //__MA_CLI_API_H__
