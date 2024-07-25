//
//  baseDefine.h
//  zsApi
//
//  Created by ���� on 2018/4/24.
//  Copyright ? 2018�� ��ɽ֤ȯ�������ι�˾. All rights reserved.
//
//----------------------------------------------------------------------------
// �޸�����      �汾          ����            ��ע
//----------------------------------------------------------------------------
// 2018-04-24    1.0           ����            �½�
//----------------------------------------------------------------------------

#if !defined(__BASE_DEFINE_H__)
#define __BASE_DEFINE_H__

#if defined (_MSC_VER) && (_MSC_VER == 1200)
#define FORMAT_LONGLONG "%I64d"
#if defined(WIN32)
typedef __int64 LONGLONG;
#endif
#else
#define FORMAT_LONGLONG "%lld"
typedef long long LONGLONG;
#endif

#if defined(WIN32) || defined(WIN64)
  #if defined(ZSAPI_EXPORTS)
    #define ZSAPI __declspec(dllexport)
  #else
    #define ZSAPI __declspec(dllimport)
  #endif
  #define ZSAPI_STDCALL __stdcall
#else
  #define ZSAPI
  #define ZSAPI_STDCALL
  #if !defined __int64
    #define __int64 long long
  #endif
#endif


//////////////////////////////////////////////////////////////////////////
// ��¼״̬����
#define LOGIN_OFFLINE                       '0'       //δ��¼
#define LOGIN_ONLINE                        '1'       //�ѵ�¼


#define ADRRESS_MAX                   128
#define USER_MAX                      16
#define ERR_MSG_SIZE                  256


#pragma pack(4)

// ����̶�����
struct STReqFix
{
    char          szUserName[USER_MAX];       // �û���
    LONGLONG      llAcctCode;                 // �ʲ��˻�
    int           iFunction;                  // ���ܴ���
    char          chOpRole;                   // �û���ɫ
    char          szSession[128];             // �Ựƾ֤
};

// ��1���������
struct STFirstSet
{
    int           iCode;                      //������
    char          szText[ERR_MSG_SIZE];       //�����ı�
};

#pragma pack()

#endif  //__BASE_DEFINE_H__