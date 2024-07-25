#if !defined(__ZS_STK_TRADE_API_STRUCT_H__)
#define __ZS_STK_TRADE_API_STRUCT_H__

#include "baseDefine.h"

//-------------------------------���Խ���ע��------------------------------------
struct STReqAcctRegister
{
    char          szUserName[16];               // �û���
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
};

struct STRspAcctRegister
{
    char          szSessionId[128];             // �Ựƾ֤
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          chLoginStatus;                // ��¼״̬
    int           iAllowStrategy;               // �Ƿ�������Խ���
};

//-------------------------------�ʲ��˻���¼------------------------------------
struct STReqTradeLogin
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          chAcctType;                   // �˻����� '0':��Ʊ '1':����
    char          szAuthData[64];               // ��֤����-����
};

struct STRspTradeLogin
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
};

//-------------------------------�û���¼------------------------------------
struct STReqLogin
{
    char          szUserName[16];               // �û���
    char          chLoginType;                  // ���� 'X':�ǳ� ��������¼
    char          szPassword[256];              // ����
};

struct STRspLogin
{
};

//-------------------------------�ɶ��˻���ѯ--------------------------
struct STReqQryHolder
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
};

struct STRspQryHolder
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szTrdacct[10 + 1];            // ֤ȯ�˻�
};

//-------------------------------ί���µ�------------------------------------
struct STReqOrder
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ�� SZ SH
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szOrderPrice[21 + 1];         // ί�м۸�
    LONGLONG      llOrderQty;                   // ί������
    int           iStkBiz;                      // ֤ȯҵ�� ��(100)��(101)
    short         iStrategySn;                  // ���Ա��
};

struct STRspOrder
{
    int           iOrderBsn;                    // ί������
    char          szOrderId[10 + 1];            // ��ͬ���
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szOrderPrice[21 + 1];         // ί�м۸�
    LONGLONG      llOrderQty;                   // ί������
    char          szOrderAmt[21 + 1];           // ί�н��
    char          szOrderFrzAmt[21 + 1];        // ������
    char          szStkbd[2 + 1];               // ���װ�� SZ SH
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szStkName[16 + 1];            // ֤ȯ����
    int           iStkBiz;                      // ֤ȯҵ�� ��(100)��(101)
    char          szTrdacct[10 + 1];            // ֤ȯ�˻�
    short         iStrategySn;                  // ���Ա��
};

//-------------------------------ί�г���------------------------------------
struct STReqCancelOrder
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��  SZ SH
    char          szOrderId[10 + 1];            // ��ͬ���
    int           iOrderBsn;                    // ί������
};

struct STRspCancelOrder
{
    int           iOrderBsn;                    // ί������
    char          szOrderId[10 + 1];            // ��ͬ���
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szOrderPrice[21 + 1];         // ί�м۸�
    LONGLONG      llOrderQty;                   // ί������
    char          szOrderAmt[21 + 1];           // ί�н��
    char          szOrderFrzAmt[21 + 1];        // ������
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szStkName[16 + 1];            // ֤ȯ����
    int           iStkBiz;                      // ֤ȯҵ�� ��(100)��(101)
    char          szTrdacct[10 + 1];            // ֤ȯ�˻�
    char          szMsgOk[32 + 1];              // �ڳ���Ϣ
    char          szCancelList[256 + 1];        // �����б�
};

//-------------------------------ί�в�ѯ--------------------------
struct STReqQryOrder
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          chQueryFlag;                  // ��ѯ����
    char          szQueryPos[32 + 1];           // ��λ��
    int           iQueryNum;                    // ��ѯ���� ���ֵ1000
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szOrderId[10 + 1];            // ��ͬ���
    int           iOrderBsn;                    // ί������
    char          szTrdacct[10 + 1];            // �����˻�
    short         iStrategySn;                  // ���Ա��
    char          chFlag;                       // ��ѯ��־ '0':��ѯ����ί����Ϣ '1':��ѯ����ί����Ϣ ����:ȫ��
};

struct STRspQryOrder
{
    char          szQryPos[32 + 1];             // ��λ��
    char          szOrderId[10 + 1];            // ��ͬ���
    char          chOrderStatus;                // ί��״̬
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szStkName[16 + 1];            // ֤ȯ����
    char          szOrderPrice[21 + 1];         // ί�м۸�
    LONGLONG      llOrderQty;                   // ί������
    char          szOrderAmt[21 + 1];           // ί�н��
    char          szOrderFrzAmt[21 + 1];        // ί�ж�����
    char          szOrderUfzAmt[21 + 1];        // ί�нⶳ���
    LONGLONG      llWithdrawnQty;               // �ѳ�������
    LONGLONG      llMatchedQty;                 // �ѳɽ�����
    char          chIsWithdraw;                 // ������־
    char          chIsWithdrawn;                // �ѳ�����־
    int           iStkBiz;                      // ֤ȯҵ�� ��(100)��(101)
    int           iOrderBsn;                    // ί������
    char          szRawOrderId[10 + 1];         // ԭ��ͬ���
    short         iStrategySn;                  // ���Ա��
    char          szOrderTime[32 + 1];          // ί��ʱ��
    char          szTrdacct[10 + 1];            // �����˻�
};

//-------------------------------�ɳ���ί�в�ѯ------------------------
struct STReqQryWithdrawableOrder
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szOrderId[10 + 1];            // ��ͬ���
    short         iStrategySn;                  // ���Ա��
    int           iOrderBsn;                    // ί������
};

struct STRspQryWithdrawableOrder
{
    char          szOrderId[10 + 1];            // ��ͬ���
    char          chOrderStatus;                // ί��״̬
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szStkName[16 + 1];            // ֤ȯ����
    LONGLONG      llOrderQty;                   // ί������
    char          szOrderPrice[21 + 1];         // ί�м۸�
    char          szOrderAmt[21 + 1];           // ί�н��
    char          szOrderFrzAmt[21 + 1];        // ί�ж�����
    char          szOrderUfzAmt[21 + 1];        // ί�нⶳ���
    char          szOrderTime[32 + 1];          // ί��ʱ��
    LONGLONG      llWithdrawnQty;               // �ѳ�������
    LONGLONG      llMatchedQty;                 // �ѳɽ�����
    char          chIsWithdraw;                 // ������־
    char          chIsWithdrawn;                // �ѳ�����־
    int           iStkBiz;                      // ֤ȯҵ��
    int           iOrderBsn;                    // ί������
    char          szMatchedAmt[21 + 1];         // �ɽ����
    char          szTrdacct[10 + 1];            // �����˻�
    short         iStrategySn;                  // ���Ա��
};

//-------------------------------�ɽ���ѯ--------------------------------
struct STReqQryFill
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szOrderId[10 + 1];            // ��ͬ���
    int           iOrderBsn;                    // ί������
    char          szTrdacct[10 + 1];            // �����˻�
    char          chQueryFlag;                  // ��ѯ����
    char          szQueryPos[32 + 1];           // ��λ��
    int           iQueryNum;                    // ��ѯ����
    short         iStrategySn;                  // ���Ա��
    char          chFlag;                       // ��ѯ��־ '0':����ί�еĳɽ� '1':����ί�еĳɽ� ����:ȫ��
};

struct STRspQryFill
{
    char          szQryPos[32 + 1];             // ��λ��
    char          szMatchedTime[8 + 1];         // �ɽ�ʱ��
    int           iOrderDate;                   // ί������
    int           iOrderSn;                     // ί�����
    int           iOrderBsn;                    // ί������
    char          szOrderId[10 + 1];            // ��ͬ���
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szTrdacct[10 + 1];            // ֤ȯ�˻�
    int           iStkBiz;                      // ֤ȯҵ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szStkName[16 + 1];            // ֤ȯ����
    char          chCurrency;                   // ���Ҵ���
    char          szBondInt[21 + 1];            // ծȯ��Ϣ
    char          szOrderPrice[21 + 1];         // ί�м۸�
    LONGLONG      llOrderQty;                   // ί������
    char          szOrderAmt[21 + 1];           // ί�н��
    char          szOrderFrzAmt[21 + 1];        // ί�ж�����
    char          szMatchedSn[16 + 1];          // �ɽ����
    char          szMatchedPrice[21 + 1];       // �ɽ��۸�
    char          szMatchedQty[21 + 1];         // �ѳɽ�����
    char          szMatchedAmt[21 + 1];         // �ѳɽ����
    char          chMatchedType;                // �ɽ�����
    char          chIsWithdraw;                 // ������־
    char          chOrderStatus;                // ί��״̬
    short         iStrategySn;                  // ���Ա��
};


//-------------------------------�ʽ��ѯ--------------------------
struct STReqQryMoney
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
};

struct STRspQryMoney
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szFundBln[21 + 1];            // �ʽ����
    char          szFundAvl[21 + 1];            // �ʽ���ý��
    char          szFundTrdFrz[21 + 1];         // �ʽ��׶�����
    char          szFundTrdOtd[21 + 1];         // �ʽ�����;���
    char          szTotalAssets[21 + 1];        // �ʲ���ֵ
    char          szFundValue[21 + 1];          // �ʽ��ʲ�
    char          szMarketValue[21 + 1];        // ��ֵ
};

//-------------------------------�ɷݲ�ѯ--------------------------------
struct STReqQryHolding
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szTrdacct[10 + 1];            // �����˻�
};

struct STRspQryHolding
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szTrdacct[10 + 1];            // �����˻�
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szStkName[16 + 1];            // ֤ȯ����
    LONGLONG      llStkBln;                     // ֤ȯ���
    LONGLONG      llStkAvl;                     // ֤ȯ��������
    LONGLONG      llStkFrz;                     // ֤ȯ��������
    LONGLONG      llStkTrdOtd;                  // ֤ȯ������;����

    char          szCostPrice[21 + 1];          // �ɱ��۸�
    char          szStkBcostRlt[21 + 1];        // ֤ȯ����ɱ���ʵʱ��
    char          szMktVal[21 + 1];             // ��ֵ
    char          szProIncome[21 + 1];          // �ο�ӯ��
    char          szProfitRate[21 + 1];         // ӯ������
    LONGLONG      llStkQty;                     // ��ǰӵ����
};

//-------------------------------���ɽ���������----------------------------
struct STReqMaxTradeQty
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szOrderPrice[21 + 1];         // ί�м۸�
    int           iStkBiz;                      // ֤ȯҵ��
    char          szTrdacct[10 + 1];            // ֤ȯ�˻�
};

struct STRspMaxTradeQty
{
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    LONGLONG      llOrderQty;                   // ί������
    char          szTrdacct[10 + 1];            // ֤ȯ�˻�
};






//-------------------------------�ɽ��ر�����------------------------------------
struct STRtnOrderFill
{
    char          szMatchedSn[16 + 1];        // �ɽ����
    char          szStkCode[8 + 1];           // ֤ȯ����
    char          szOrderId[10 + 1];          // ��ͬ���
    char          szTrdacct[16 + 1];          // �����˻�
    LONGLONG      llMatchedQty;               // ���γɽ�����
    char          szMatchedPrice[11 + 1];     // ���γɽ��۸�
    char          szOrderFrzAmt[21 + 1];      // ί�ж�����
    char          szRltSettAmt[21 + 1];       // ʵʱ������
    char          szFundAvl[21 + 1];          // �ʽ���ý��ɽ���
    LONGLONG      llStkAvl;                   // ֤ȯ�����������ɽ���
    int           iMatchedDate;               // �ɽ�����
    char          szMatchedTime[8 + 1];       // �ɽ�ʱ��
    char          chIsWithdraw;               // ������־ 'F':���� 'T':����
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    int           iOrderBsn;                  // ί������
    short         iStrategySn;                // ���Ա��
    char          szStkbd[2 + 1];             // ���װ��
    char          chMatchedType;              // �ɽ����� '1':�Ƿ�ί�г����ɽ� '2':�������������ϳɽ�
    char          chOrderStatus;              // ί��״̬ '0':δ�� '2':�ѱ� '6':�ѳ� '8':�ѳ� '9':�ϵ�
    int           iStkBiz;                    // ֤ȯҵ��
    char          szOfferRetMsg[64 + 1];      // �걨��Ϣ �������ϵ�ʱ���طϵ�ԭ��
    LONGLONG      llOrderQty;                 // ί������
    LONGLONG      llWithdrawnQty;             // �ѳ�������
    LONGLONG      llTotalMatchedQty;          // �ۼƳɽ�����
    char          szTotalMatchedAmt[21 + 1];  // �ۼƳɽ����
    LONGLONG      llStkQty;                   // ӵ����
    char          szMatchedAmt[21 + 1];       // �ѳɽ����
};

//-------------------------------ȷ�ϻر�����------------------------------------
struct STRtnOrderConfirm
{
    char          szStkCode[8 + 1];           // ֤ȯ����
    char          szOrderId[10 + 1];          // ��ͬ���
    char          szTrdacct[16 + 1];          // �����˻�
    char          chIsWithdraw;               // ������־ 'F':���� 'T':����
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    int           iOrderBsn;                  // ί������
    short         iStrategySn;                // ���Ա��
    char          szStkbd[2 + 1];             // ���װ��
    char          chOrderStatus;              // ί��״̬ '0':δ�� '2':�ѱ� '6':�ѳ� '8':�ѳ� '9':�ϵ�
    int           iStkBiz;                    // ֤ȯҵ��
    int           iOrderDate;                 // ί������
    char          szOrderPrice[21 + 1];       // ί�м۸�
    LONGLONG      llOrderQty;                 // ί������
};




//-------------------------------������ȯ���ȯ��Ϣ��ѯ------------------------------
struct STReqQryUndlStkInfo
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��  
    char          szStkCode[8 + 1];             // ֤ȯ����  
    char          chCurrEnableFi;               // �������ʱ�־ '0':���� '1':������
    char          chCurrEnableSl;               // ������ȯ��־ '0':���� '1':������ 
    char          chQueryFlag;                  // ��ѯ����  
    char          szQueryPos[32 + 1];           // ��λ��
    int           iQueryNum;                    // ��ѯ����
};

struct STRspQryUndlStkInfo
{
    char          szQryPos[32 + 1];             // ��λ��        
    char          szStkbd[2 + 1];               // ���װ��      
    char          szStkCode[8 + 1];             // ֤ȯ����      
    char          szStkName[16 + 1];            // ֤ȯ����      
    char          szFiMarginRatio[21 + 1];      // ���ʱ�֤�����
    char          szSlMarginRatio[21 + 1];      // ��ȯ��֤�����
    char          chCurrEnableFi;               // �������ʱ�־ '0':���� '1':������
    char          chCurrEnableSl;               // ������ȯ��־ '0':���� '1':������
};

//-------------------------------������ȯ����֤ȯ��Ϣ��ѯ----------------------------
struct STReqQryColStkInfo
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��  
    char          szStkCode[8 + 1];             // ֤ȯ����  
    char          chQueryFlag;                  // ��ѯ����  
    char          szQueryPos[32 + 1];           // ��λ��    
    int           iQueryNum;                    // ��ѯ����  
};

struct STRspQryColStkInfo
{
    char          szQryPos[32 + 1];             // ��λ��       
    char          szStkbd[2 + 1];               // ���װ��    
    char          szStkCode[8 + 1];             // ֤ȯ����    
    char          szStkName[16 + 1];            // ֤ȯ����    
    char          szCollatRatio[21 + 1];        // ����Ʒ������
    char          chCreditFundUseFlag;          // �����ʽ�ʹ�ñ�־
};

//-------------------------------������ȯί��------------------------------------
struct STReqOrderCredit
{   
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ�� SZ SH
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szOrderPrice[21 + 1];         // ί�м۸�
    LONGLONG      llOrderQty;                   // ί������
    int           iStkBiz;                      // ֤ȯҵ�� ������(702)��ȯ��(703)
    short         iStrategySn;                  // ���Ա��
    char          szRepayOrderId[10 + 1];       // ������ͬ���
    int           iRepayOpeningDate;            // ������Լ����
    char          szRepayStkCode[8 + 1];        // ����֤ȯ����
};

struct STRspOrderCredit
{
    int           iOrderBsn;                    // ί������
    char          szOrderId[10 + 1];            // ��ͬ���
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szOrderPrice[21 + 1];         // ί�м۸�
    LONGLONG      llOrderQty;                   // ί������
    char          szOrderAmt[21 + 1];           // ί�н��
    char          szOrderFrzAmt[21 + 1];        // ������
    char          szStkbd[2 + 1];               // ���װ�� SZ SH
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szStkName[16 + 1];            // ֤ȯ����
    int           iStkBiz;                      // ֤ȯҵ�� ������(702)��ȯ��(703)
    char          szTrdacct[10 + 1];            // ֤ȯ�˻�
    short         iStrategySn;                  // ���Ա��
};

//-------------------------------������ȯֱ�ӻ���----------------------------
struct STReqRepay
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          chRepayType;                  // �������� '0':��������Ƿ�� '1':������ȯ����
    char          szRepayOrderId[10 + 1];       // ������ͬ���
    int           iRepayOpeningDate;            // ������Լ����
    char          szRepayStkCode[8 + 1];        // ����֤ȯ����
    char          szRepayContractAmt[21 + 1];   // �������
    char          chRepayAmtCls;                // ���������� '0':ȫ���黹 '1':���黹���� '2':���黹��Ϣ
    char          szRemark[128 + 1];            // ��ע
};

struct STRspRepay
{
    char          szRealRepayAmt[21 + 1];       // ʵ�ʻ�����
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szRepayContractAmt[21 + 1];   // �������
};

//-------------------------------������ȯ�ֲֲ�ѯ--------------------------------
struct STReqQryHoldingCredit
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szTrdacct[10 + 1];            // �����˻�
    char          chContractFlag;               // ���ú�Լ����
    char          chBizFlag;                    // ҵ���־ '0':��ǰ�˻��ɷ� '1':��ǰ�����˻���Ӧ����ͨ�˻��ɷ�
    
};

struct STRspQryHoldingCredit
{
    char          szQryPos[32 + 1];             // ��λ��                    
    LONGLONG      llCuacctCode;                 // �ʲ��˻�            
    char          szStkbd[2 + 1];               // ���װ��            
    char          szTrdacct[10 + 1];            // �����˻�          
    char          szStkCode[8 + 1];             // ֤ȯ����            
    char          szStkName[16 + 1];            // ֤ȯ����                       
    LONGLONG      llStkPrebln;                  // ֤ȯ�������        
    LONGLONG      llStkBln;                     // ֤ȯ���            
    LONGLONG      llStkAvl;                     // ֤ȯ��������        
    LONGLONG      llStkFrz;                     // ֤ȯ��������        
    LONGLONG      llStkUfz;                     // ֤ȯ�ⶳ����        
    LONGLONG      llStkTrdFrz;                  // ֤ȯ���׶�������    
    LONGLONG      llStkTrdUfz;                  // ֤ȯ���׽ⶳ����    
    LONGLONG      llStkTrdOtd;                  // ֤ȯ������;����    
    LONGLONG      llStkTrdBln;                  // ֤ȯ������������
    LONGLONG      llStkQty;                     // ��ǰӵ����          
    LONGLONG      llStkRemain;                  // ��ȯ��������        
    LONGLONG      llStkSale;                    // ������������
    char          chIsCollat;                   // �Ƿ��ǵ���Ʒ
    char          szCollatRatio[21 + 1];        // ����Ʒ������
    LONGLONG      llMktQty;                     // ��ǰӵ�������˻���
    char          szAveragePrice[21 + 1];       // �������
};

//-------------------------------������ȯ��Լ��ѯ----------------------------
struct STReqQryContract
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    int           iBgnDate;                     // ��ʼ����
    int           iEndDate;                     // ��������
    char          szOrderId[10 + 1];            // ��ͬ���
    char          chContractType;               // ��Լ���� '0':���� '1':��ȯ
    char          szQueryPos[32 + 1];           // ��λ��  
    int           iQueryNum;                    // ��ѯ����
    char          szStkbd[2 + 1];               // ���װ��
    char          szTrdacct[10 + 1];            // �����˻�
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          chContractStatus;             // ��Լ״̬ '0':����δ�黹 '1':���ֹ黹 '2':����δƽ�� '3':�����˽� '4':�ֹ��˽� '5':ʵʱ��Լ '6':չ���� 
    char          chRepayFlag;                  // ƽ��״̬ '0':ȫ�� '1':δƽ�� '2':��ƽ��
    char          chQueryFlag;                  // ��ѯ����
};

struct STRspQryContract
{
    char          szQryPos[32 + 1];             // ��λ��
    LONGLONG      llCashNo;                     // ͷ����            
    int           iTrdDate;                     // ��������            
    char          chContractType;               // ��Լ���� '0':���� '1':��ȯ
    char          szTrdacct[10 + 1];            // �����˻�            
    char          szStkbd[2 + 1];               // ���װ��
    int           iOpeningDate;                 // ��������
    char          szStkCode[8 + 1];             // ֤ȯ����
    char          szOrderId[10 + 1];            // ��ͬ���
    char          szFiDebtsAmt[21 + 1];         // ���ʸ�ծ���        
    LONGLONG      llSlDebtsQty;                 // ��ȯ��ծ����        
    LONGLONG      llRepaidQty;                  // ��ȯ�ѻ�����        
    char          szRepaidAmt[21 + 1];          // �����ѻ����        
    char          chContractStatus;             // ��Լ״̬ '0':����δ�黹 '1':���ֹ黹 '2':��Լ�ѹ��� '3':�����˽� '4':�ֹ��˽� '5':ʵʱ��Լ '6':չ���� 
    int           iContractExpireDate;          // ��Լ������          
    char          szMarginRatio[21 + 1];        // ��֤�����          
    char          szMarginAmt[21 + 1];          // ռ�ñ�֤��          
    char          szRights[21 + 1];             // δ����Ȩ����      
    LONGLONG      llRightsQty;                  // δ����Ȩ������      
    char          szOverdueFee[21 + 1];         // ����δ����Ϣ��      
    int           iLastRepayDate;               // ��󳥻�����        
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szOrderPrice[21 + 1];         // ί�м۸�
    LONGLONG      llOrderQty;                   // ί������
    char          szOrderAmt[21 + 1];           // ί�н��
    char          szOrderFrzAmt[21 + 1];        // ί�ж�����        
    LONGLONG      llWithdrawnQty;               // �ѳ�������          
    LONGLONG      llMatchedQty;                 // �ɽ�����            
    char          szMatchedAmt[21 + 1];         // �ɽ����            
    char          szRltSettAmt[21 + 1];         // ʵʱ������        
    char          szSlDebtsMktvalue[21 + 1];    // ��ȯ��ծ��ֵ        
    LONGLONG      llRltRepaidQty;               // ��ȯʵʱ�黹����    
    char          szRltRepaidAmt[21 + 1];       // ����ʵʱ�黹���    
    char          szMatchedAmtRepay[21 + 1];    // ���ɽ����          
    int           iCalIntDate;                  // ��ʼ��Ϣ����        
    char          szContractInt[21 + 1];        // ��Լ��Ϣ            
    char          szContractIntAccrual[21 + 1]; // ��Ϣ����            
    char          szOverRights[21 + 1];         // ����δ����Ȩ��      
    char          szRightsRepay[21 + 1];        // ������Ȩ��          
    char          szRightsRlt[21 + 1];          // ʵʱ����Ȩ��        
    char          szOverRightsRlt[21 + 1];      // ʵʱ����Ԥ��Ȩ��    
    LONGLONG      llOverRightsQty;              // ����δ����Ȩ������  
    LONGLONG      llRightsQtyRepay;             // �ѳ���Ȩ������      
    LONGLONG      llRightsQtyRlt;               // ʵʱ����Ȩ������    
    LONGLONG      llOverRightsQtyRlt;           // ʵʱ��������Ȩ������
    char          szContractFee[21 + 1];        // ������ȯϢ��        
    char          szFeeRepay[21 + 1];           // ������Ϣ��          
    char          szFeeRlt[21 + 1];             // ʵʱ����Ϣ��        
    char          szOverDuefeeRlt[21 + 1];      // ʵʱ��������Ϣ��    
    char          szPuniDebts[21 + 1];          // ���ڱ���Ϣ        
    char          szPuniDebtsRepay[21 + 1];     // ����Ϣ����        
    char          szPuniDebtsRlt[21 + 1];       // ʵʱ���ڱ���Ϣ    
    char          szPuniFee[21 + 1];            // ��Ϣ�����ķ�Ϣ      
    char          szPuniFeeRepay[21 + 1];       // ��������Ϣ          
    char          szPuniFeeRlt[21 + 1];         // ʵʱ����Ϣ�ѷ�Ϣ    
    char          szPuniRights[21 + 1];         // ����Ȩ�淣Ϣ        
    char          szPuniRightsRepay[21 + 1];    // Ȩ�淣Ϣ����        
    char          szPuniRightsRlt[21 + 1];      // ʵʱ����Ȩ�淣Ϣ    
    int           iClosingDate;                 // ��Լ�˽�����        
    char          szClosingPrice[21 + 1];       // ��Լ�˽�۸�        
    char          chContractCls;                // ��Լ��� '0':ʵʱ��Լ '1':�ѿ��ֺ�Լ
    char          szProIncome[21 + 1];          // �ο�ӯ��
    char          szPledgeCuacct[21 + 1];       // ��Ѻ�ʲ�
    char          szFiRepayAmt[21 + 1];         // ���ʳ���
    LONGLONG      llSlRepayQty;                 // ��ȯ����
};

//-------------------------------���ÿͻ��ʲ���ծ��ѯ------------------------
struct STReqQryCustDebts
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
};

struct STRspQryCustDebts
{      
    LONGLONG      llCuacctCode;                 // �ʲ��˻�              
    char          szFiRate[21 + 1];             // ��������        
    char          szSlRate[21 + 1];             // ��ȯ����        
    char          szFreeIntRate[21 + 1];        // ��Ϣ����        
    char          chCreditStatus;               // ����״̬ '0':���� '1':δ���� '2':������ '3':���� '9':����
    char          szMarginRate[21 + 1];         // ά�ֵ�������    
    char          szRealRate[21 + 1];           // ʵʱ��������    
    char          szTotalAssert[21 + 1];        // ���ʲ�          
    char          szTotalDebts[21 + 1];         // �ܸ�ծ          
    char          szMarginValue[21 + 1];        // ��֤��������  
    char          szFundAvl[21 + 1];            // �ʽ���ý��    
    char          szFundBln[21 + 1];            // �ʽ����        
    char          szSlAmt[21 + 1];              // ��ȯ���������ʽ�
    char          szGuaranteOut[21 + 1];        // ��ת�������ʲ�  
    char          szColMktVal[21 + 1];          // ����֤ȯ��ֵ    
    char          szFiAmt[21 + 1];              // ���ʱ���        
    char          szTotalFiFee[21 + 1];         // ����Ϣ��        
    char          szFiTotalDebts[21 + 1];       // ���ʸ�ծ�ϼ�    
    char          szSlMktVal[21 + 1];           // Ӧ����ȯ��ֵ    
    char          szTotalSlFee[21 + 1];         // ��ȯϢ��        
    char          szSlTotalDebts[21 + 1];       // ��ȯ��ծ�ϼ�    
    char          szFiCredit[21 + 1];           // �������Ŷ��    
    char          szFiCreditAvl[21 + 1];        // ���ʿ��ö��    
    char          szFiCreditFrz[21 + 1];        // ���ʶ�ȶ���    
    char          szSlCredit[21 + 1];           // ��ȯ���Ŷ��    
    char          szSlCreditAvl[21 + 1];        // ��ȯ���ö��    
    char          szSlCreditFrz[21 + 1];        // ��ȯ��ȶ���    
    char          szRights[21 + 1];             // ����Ȩ��        
    char          szRightsUncomer[21 + 1];      // ����Ȩ�棨��;��
    LONGLONG      llRightsQty;                  // ���Ȩ��        
    LONGLONG      llRightsQtyUncomer;           // ���Ȩ�棨��;��
    char          szTotalCredit[21 + 1];        // �ܶ��          
    char          szTotalCteditAvl[21 + 1];     // �ܿ��ö��      
};

//-------------------------------֤ȯ��ֵ��Ȳ�ѯ------------------------------------
struct STReqQryMktQuota
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
};

struct STRspQryMktQuota
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szTrdacct[10 + 1];            // ֤ȯ�˻�
    LONGLONG      llMktQuota;                   // ��ֵ���
    LONGLONG      llKcbMktQuota;                // ��ֵ���(�ƴ���) 
};

//-------------------------------�����¹���Ϣ��ѯ------------------------------------
struct STReqQryIpoInfo
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
};

struct STRspQryIpoInfo
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szStkbd[2 + 1];               // ���װ��
    char          szStkCode[8 + 1];             // �깺����
    char          szStkName[16 + 1];            // �깺��������
    char          szLinkStk[8 + 1];             // ���ɴ���
    char          chIssueType;                  // ���з�ʽ  '0':�깺 '1':����
    int           iIssueDate;                   // �깺����
    char          szFixPrice[21 + 1];           // �깺�۸�
    int           iBuyUnit;                     // �깺��λ
    int           iMinQty;                      // �깺����
    int           iMaxQty;                      // �깺����
};

//ת������ҵ����Ϣ��ѯ
struct STReqQryBankInfo
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          chCurrency;                   // ���Ҵ���
    char          szBankCode[4 + 1];            // ���д���
};
struct STRspQryBankInfo
{
    char          szBankCode[4 + 1];            // ���д���
    char          chSourceType;                 // ����    
    char          chBankTranType;               // ת�ʷ�ʽ  
    char          chFundPwdFlag;                // �ʽ�����У��
    char          chBankPwdFlag;                // ��������У��
    char          chCheckBankId;                // �����ʺ�У��
    char          chCheckIdNo;                  // ֤��У���־  
    char          szOrgId[4 + 1];               // ��������
    char          chCurrency;                   // ���Ҵ���
    char          chStatus;                     // ����״̬
};

//��֤ת��
struct STReqBankStkTrans
{
    LONGLONG  llCuacctCode;                      //�ʽ��˺�
    char      chCurrency;                        //���Ҵ���
    char      szFundPwd[32 + 1];                 //�ʽ�����
    char      szBankCode[4 + 1];                 //���д���
    char      szBankPwd[32 + 1];                 //��������
    char      chBankTranType;                    //ת������ '1':����ת֤ȯ '2':֤ȯת����
    char      szTransAmt[21 + 1];                //ת�ʽ��                          
};
struct STRspBankStkTrans
{
    int  iSNo;                                  //ί�����
    int  iSysErrorId;                           //�������
    char szErrorMsg[64 + 1];                    //������Ϣ

};

//��ѯ�����˻����
struct STReqQryBankBalance
{
    LONGLONG      llCuacctCode;                 // �ʽ��˺�
    char          chCurrency;                   // ���Ҵ���
    char          szFundPwd[32 + 1];            // �ʽ�����
    char          szBankCode[4 + 1];            // ���д���
    char          szBankPwd[32 + 1];            // ��������
};
struct STRspQryBankBalance
{
    int           iSNo;                         // ί�����
    char          szErrorMsg[64 + 1];           // ������Ϣ
    int           iSysErrId;                    // �������
    char          szFundEffect[21 + 1];         // �������
};

//��֤ת�ʲ�ѯ
struct STReqQryBankStkTransInfo
{
    LONGLONG      llCuacctCode;                 // �ʽ��˺�
    char          chCurrency;                   // ���Ҵ���
    int           iSNo;                         // ί�����
};
struct STRspQryBankStkTransInfo
{
    int           iOperDate;                    // ת������
    int           iOperTime;                    // ת��ʱ��
    LONGLONG      llCuacctCode;                 // �ʽ��˺�
    char          chCurrency;                   // ���Ҵ���
    char          szBankCode[4 + 1];            // ���д���
    char          chBankTranId;                 // ҵ������
    int           iSNo;                         // ί�����
    char          szFundEffect[21 + 1];         // ί�н��
    char          szFundBal[21 + 1];            // ���
    char          szRemark[32 + 1];             // ��ע��Ϣ
    char          chStatus;                     // ������ 
    char          chSourceType;                 // ������ 
    char          szBankMsgId[16 + 1];          // �ⲿ��Ϣ����
    char          szBankMsg[64 + 1];            // �ⲿ��Ϣ����
    char          szErrorMsg[64 + 1];           // ϵͳ������Ϣ
    int           iSysErrId;                    // ϵͳ�������
};

//�޸Ľ�������
struct STReqModifyTradePwd
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          szNewPwd[32 + 1];             // ������
};
struct STRspModifyTradePwd
{
    char          szMsgOk[32 + 1];              // �ɹ���Ϣ
};

//�޸��ʽ�����
struct STReqModifyFundPwd
{
    LONGLONG      llCuacctCode;                 // �ʽ��ʻ�
    char          szOldFundPwd[32 + 1];         // ���ʽ�����
    char          szNewFundPwd[32 + 1];         // ���ʽ�����
};
struct STRspModifyFundPwd
{
    char          szMsgOk[32 + 1];              // �ɹ���Ϣ
};

//���н��׹�̨�ʽ��ѯ
struct STReqQryRpcFund
{
    LONGLONG      llCuacctCode;                 // �ʲ��˻�
    char          chCurrency;                   // ���Ҵ���
};
struct STRspQryRpcFund
{
    LONGLONG      llCuacctCode;                 // �ʽ��˻�
    char          szOrgId[4 + 1];               // ��������
    char          chCurrency;                   // ���Ҵ���
    char          szFundBal[21 + 1];            // �ʽ����
    char          szFundAvl[21 + 1];            // �ʽ���ý��
    char          szMarketValue[21 + 1];        // �ʲ���ֵ
    char          szFund[21 + 1];               // �ʽ��ʲ�
    char          szStkValue[21 + 1];           // ��ֵ
    int           iFundSeq;                     // ���ʽ��־
};

//�ʽ𻮲�
struct STReqFundTransfer
{
    char         szOrgId[4 + 1];                // ��������
    LONGLONG     llCuacctCode;                  // �ʽ��˺�
    char         chCurrency;                    // ���Ҵ���
    char         szFundAvl[21 + 1];             // �ʽ����
    char         chDirect;                      // �������� '0':��ͨ�ڵ㻮�롢VIP ϵͳ���� '1':��ͨ�ڵ㻮����VIP ϵͳ����
};
struct STRspFundTransfer
{
    int          iSNo;                          // ������ˮ��
    int          iVipSno;                       // VIP������ˮ��
};

//��֤ת�������˺Ų�ѯ
struct STReqQryBankAcct
{
    char          szBankCode[4 + 1];            // ���д���
    char          chCurrency;                   // ���Ҵ���
    LONGLONG      llCuacctCode;                 // �ʽ��˺�
};
struct STRspQryBankAcct
{
    char          szOrgId[4 + 1];                // ��������
    char          szBankCode[4 + 1];             // ���д���
    char          szBankName[32 + 1];            // ��������
    char          chCurrency;                    // ���Ҵ���
    char          szBankId[32 + 1];              // �����ʻ�
    LONGLONG      llCuacctCode;                  // �ʽ��ʺ�
    char          chLinkFlag;                    // ת�ʱ�ʶ
    char          chSourceType;                  // ������
};



//-------------------------------��Ȩί���걨------------------------------------
struct STReqOrderOpt
{
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szStkbd[2 + 1];             // ���װ��
    char          szOptNum[16 + 1];           // ��Լ����
    char          szOrderPrice[11 + 1];       // ί�м۸�
    LONGLONG      llOrderQty;                 // ί������
    int           iStkBiz;                    // ֤ȯҵ��
    int           iStkBizAction;              // ֤ȯҵ����Ϊ
    short         iStrategySn;                // ���Ա��
};

struct STRspOrderOpt
{
    int           iOrderBsn;                  // ί������
    char          szOrderId[10 + 1];          // ��ͬ���
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szOrderPrice[11 + 1];       // ί�м۸�
    LONGLONG      llOrderQty;                 // ί������
    char          szOrderAmt[21 + 1];         // ί�н��
    char          szOrderFrzAmt[21 + 1];      // ί�ж�����
    char          szStkbd[2 + 1];             // ���װ��
    char          szTrdacct[10 + 1];          // ֤ȯ�˻�
    char          szSubacctCode[8 + 1];       // ֤ȯ�˻��ӱ���
    char          szOptTrdacct[18 + 1];       // ��Ȩ��Լ�˻�
    char          szOptNum[16 + 1];           // ��Լ����
    char          szOptCode[32 + 1];          // ��Լ����
    char          szOptName[32 + 1];          // ��Լ���
    char          szOptUndlCode[8 + 1];       // ���֤ȯ����
    char          szOptUndlName[16 + 1];      // ���֤ȯ����
    int           iStkBiz;                    // ֤ȯҵ��
    int           iStkBizAction;              // ֤ȯҵ����Ϊ
    short         iStrategySn;                // ���Ա��
};

//-------------------------------��Ȩί�г���------------------------------------
struct STReqOptCancelOrder
{
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szStkbd[2 + 1];             // ���װ��
    char          szOrderId[10 + 1];          // ��ͬ���
    int           iOrderBsn;                  // ί������
};

struct STRspOptCancelOrder
{
    int           iOrderBsn;                  // ί������
    char          szOrderId[10 + 1];          // ��ͬ���
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szOrderPrice[11 + 1];       // ί�м۸�
    LONGLONG      llOrderQty;                 // ί������
    char          szOrderAmt[21 + 1];         // ί�н��
    char          szOrderFrzAmt[21 + 1];      // ί�ж�����
    char          szStkbd[2 + 1];             // ���װ��
    char          szTrdacct[10 + 1];          // ֤ȯ�˻�
    char          szSubacctCode[8 + 1];       // ֤ȯ�˻��ӱ���
    char          szOptTrdacct[18 + 1];       // ��Ȩ��Լ�˻�
    char          szStkCode[32 + 1];          // ֤ȯ����
    char          szStkName[32 + 1];          // ֤ȯ����
    int           iStkBiz;                    // ֤ȯҵ��
    int           iStkBizAction;              // ֤ȯҵ����Ϊ
    char          chCancelStatus;             // �ڲ�������־ 1:�ڲ����� ��1:��ͨ����
    short         iStrategySn;                // ���Ա��
};


//-------------------------------��Ȩ�����ʽ��ѯ--------------------------
struct STReqOptQryMoney
{
    LONGLONG        llCuacctCode;               // �ʲ��˻� 
};

struct STRspOptQryMoney
{
    LONGLONG        llCuacctCode;               // �ʲ��˻� 
    char            chCurrency;                 // ���Ҵ���
    char            szMarketValue[21 + 1];      // �ʲ���ֵ �ͻ��ʲ��ܶʵʱ��
    char            szFundValue[21 + 1];        // �ʽ��ʲ� �ʽ��ʲ��ܶ�
    char            szStkValue[21 + 1];         // ��ֵ ���ʽ��ʲ��ܶ� = ��ֵ
    char            szFundPrebln[21 + 1];       // �ʽ�������� 
    char            szFundBln[21 + 1];          // �ʽ���� 
    char            szFundAvl[21 + 1];          // �ʽ���ý�� 
    char            szFundFrz[21 + 1];          // �ʽ𶳽��� 
    char            szFundUfz[21 + 1];          // �ʽ�ⶳ��� 
    char            szFundTrdFrz[21 + 1];       // �ʽ��׶����� 
    char            szFundTrdUfz[21 + 1];       // �ʽ��׽ⶳ��� 
    char            szFundTrdOtd[21 + 1];       // �ʽ�����;��� 
    char            szFundTrdBln[21 + 1];       // �ʽ��������� 
    char            chFundStatus;               // �ʽ�״̬
    char            szMarginUsed[21 + 1];       // ռ�ñ�֤�� 
    char            szMarginInclRlt[21 + 1];    // ��ռ�ñ�֤��(��δ�ɽ�) ������������ί��δ�ɽ�����ı�֤��(��ǰ����ۼ���)
    char            szFundExeMargin[21 + 1];    // ��Ȩ������֤�� 
    char            szFundExeFrz[21 + 1];       // ��Ȩ�ʽ𶳽��� 
    char            szFundFeeFrz[21 + 1];       // �ʽ���ö����� 
    char            szPaylater[21 + 1];         // �渶�ʽ� 
    char            szPreadvaPay[21 + 1];       // Ԥ�Ƶ��ʽ�� ����ETF��ȨE+1��Ԥ����ʹ��
    char            szExpPenInt[21 + 1];        // Ԥ�Ƹ�ծ��Ϣ 
    char            szFundDraw[21 + 1];         // �ʽ��ȡ��� 
    char            szFundAvlRlt[21 + 1];       // �ʽ�̬���� 
    char            szMarginInclDyn[21 + 1];    // ��̬ռ�ñ�֤��(��δ�ɽ�) ������������ί��δ�ɽ�����ı�֤��(��ʵʱ�۸����)
    char            szDailyInAmt[21 + 1];       // ������� 
    char            szDailyOutAmt[21 + 1];      // ���ճ��� 
    char            szFundRealAvl[21 + 1];      // �ʽ�ʵ�ʿ��� ����֤��Ʊ��Ȩ��̨ϵͳ���ö�̬���ù���ʱ���ʽ�ʵ�ʿ���=min���ʽ���ý��ʽ�̬���ã����������ö�̬����ʱ���ʽ�ʵ�ʿ���=�ʽ���ý��
};

//-------------------------------��Ȩ�ֲֲ�ѯ------------------------------------
struct STReqOptQryHolding
{
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szStkbd[2 + 1];             // ���װ��
    char          szTrdacct[10 + 1];          // �����˻�
    char          szOptNum[16 + 1];           // ��Լ����
    char          szOptUndlCode[8 + 1];       // ���֤ȯ����
    char          chOptSide;                  // �ֲַ��� L-Ȩ���֣�S-����֣�C-���Ҳ��Գֲ�
    char          chOptCvdFlag;               // ���ұ�־ 0-�Ǳ��Һ�Լ 1-���Һ�Լ
    char          chQueryFlag;                // ��ѯ���� 0:���ȡ���� 1:��ǰȡ���� ����ȫ������
};

struct STRspOptQryHolding
{
    LONGLONG        llCuacctCode;               // �ʲ��˻� 
    char            szStkbd[2 + 1];             // ���װ��
    char            szTrdacct[10 + 1];          // ֤ȯ�˻� 
    char            szSubacctCode[8 + 1];       // ֤ȯ�˻��ӱ��� 
    char            szOptTrdacct[18 + 1];       // ��Ȩ��Լ�˻� 
    char            chCurrency;                 // ���Ҵ��� 
    char            szOptNum[16 + 1];           // ��Լ���� 
    char            szOptCode[32 + 1];          // ��Լ���� 
    char            szOptName[32 + 1];          // ��Լ��� 
    char            chOptType;                  // ��Լ���� �ֵ�[OPT_TYPE]
    char            chOptSide;                  // �ֲַ��� 
    char            chOptCvdFlag;               // ���ұ�־ 0-�Ǳ��Һ�Լ 1-���Һ�Լ
    LONGLONG        llOptPrebln;                // ��Լ������� 
    LONGLONG        llOptBln;                   // ��Լ��� 
    LONGLONG        llOptAvl;                   // ��Լ�������� 
    LONGLONG        llOptFrz;                   // ��Լ�������� 
    LONGLONG        llOptUfz;                   // ��Լ�ⶳ���� 
    LONGLONG        llOptTrdFrz;                // ��Լ���׶������� 
    LONGLONG        llOptTrdUfz;                // ��Լ���׽ⶳ���� 
    LONGLONG        llOptTrdOtd;                // ��Լ������;���� 
    LONGLONG        llOptTrdBln;                // ��Լ������������ 
    LONGLONG        llOptClrFrz;                // ��Լ���㶳������ 
    LONGLONG        llOptClrUfz;                // ��Լ����ⶳ���� 
    LONGLONG        llOptClrOtd;                // ��Լ������;���� 
    char            szOptBcost[21 + 1];         // ��Լ����ɱ� 
    char            szOptBcostRlt[21 + 1];      // ��Լ����ɱ���ʵʱ�� 
    char            szOptPlamt[21 + 1];         // ��Լӯ����� 
    char            szOptPlamtRlt[21 + 1];      // ��Լӯ����ʵʱ�� 
    char            szOptMktVal[21 + 1];        // ��Լ��ֵ 
    char            szOptPremium[21 + 1];       // Ȩ���� 
    char            szOptMargin[21 + 1];        // ��֤�� 
    LONGLONG        llOptCvdAsset;              // ���ҹɷ����� 
    char            szOptClsProfit[21 + 1];     // ����ƽ��ӯ�� 
    char            szSumClsProfit[21 + 1];     // �ۼ�ƽ��ӯ�� 
    char            szOptFloatProfit[21 + 1];   // ����ӯ�� ����ӯ��=֤ȯ��ֵ-����ɱ�
    char            szTotalProfit[21 + 1];      // ��ӯ�� 
    LONGLONG        llOptRealPosi;              // ��Լʵ�ʳֲ� 
    LONGLONG        llOptClsUnmatched;          // ��Լƽ�ֹҵ����� ��ƽ��ί��δ�ɽ�����
    LONGLONG        llOptDailyOpenRlt;          // �����ۼƿ������� 
    char            szOptUndlCode[8 + 1];       // ���֤ȯ���� 
    char            szExerciseVal[21 + 1];      // ��Ȩ��ֵ �Ϲ�Ȩ���ֵ���Ȩ��ֵ��MAX((���-��Ȩ��), 0) * ��Լ��λ * ��Լ���� �Ϲ�Ȩ���ֵ���Ȩ��ֵ��MAX((��Ȩ��-��ļ�), 0) * ��Լ��λ * ��Լ����
    LONGLONG        llCombedQty;                // ����Ϻ�Լ���� ������ϵ���Ȩ��Լ�ֲ�����
    char            szCostPrice[22 + 1];        // ��Լ�ɱ��� 
};


//-------------------------------��Ȩ����ί�в�ѯ------------------------------------
struct STReqOptQryOrder
{
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szStkbd[2 + 1];             // ���װ��
    char          szTrdacct[10 + 1];          // �����˻�
    char          szOptNum[32 + 1];           // ��Լ����
    char          szOptUndlCode[8 + 1];       // ���֤ȯ����
    char          szCombStraCode[16 + 1];     // ��ϲ��Դ���
    char          szOrderId[10 + 1];          // ��ͬ���
    int           iOrderBsn;                  // ί������ 
    char          chQueryFlag;                // ��ѯ���� 0:���ȡ���� 1:��ǰȡ���� ����ȫ������
    char          szQryPos[32 + 1];           // ��λ��
    int           iQryNum;                    // ��ѯ����
};

struct STRspOptQryOrder
{
    char          szQryPos[32 + 1];           // ��λ��
    int           iTrdDate;                   // ��������
    int           iOrderDate;                 // ί������
    char          szOrderTime[25 + 1];        // ί��ʱ��
    int           iOrderBsn;                  // ί������
    char          szOrderId[10 + 1];          // ��ͬ���
    char          chOrderStatus;              // ί��״̬
    char          chOrderValidFlag;           // ί����Ч��־
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szStkbd[2 + 1];             // ���װ��
    char          szTrdacct[10 + 1];          // ֤ȯ�˻�
    char          szSubacctCode[8 + 1];       // ֤ȯ�˻��ӱ���
    char          szOptTrdacct[18 + 1];       // ��Ȩ��Լ�˻�
    int           iStkBiz;                    // ֤ȯҵ��
    int           iStkBizAction;              // ֤ȯҵ����Ϊ
    char          szOwnerType[3 + 1];         // ������������
    char          szOptNum[16 + 1];           // ��Լ����
    char          szOptCode[32 + 1];          // ��Լ����
    char          szOptName[32 + 1];          // ��Լ���
    char          szCombNum[16 + 1];          // ��ϱ���
    char          szCombStraCode[16 + 1];     // ��ϲ��Դ���
    char          szLeg1Num[16 + 1];          // �ɷ�һ��Լ����
    char          szLeg2Num[16 + 1];          // �ɷֶ���Լ����
    char          szLeg3Num[16 + 1];          // �ɷ�����Լ����
    char          szLeg4Num[16 + 1];          // �ɷ��ĺ�Լ����
    char          chCurrency;                 // ���Ҵ���
    char          szOrderPrice[11 + 1];       // ί�м۸�
    LONGLONG      llOrderQty;                 // ί������
    char          szOrderAmt[21 + 1];         // ί�н��
    char          szOrderFrzAmt[21 + 1];      // ί�ж�����
    char          szOrderUfzAmt[21 + 1];      // ί�нⶳ���
    LONGLONG      llOfferQty;                 // �걨����
    int           iOfferStime;                // �걨ʱ��
    LONGLONG      llWithdrawnQty;             // �ѳ�������
    LONGLONG      llMatchedQty;               // �ѳɽ�����
    char          szMatchedAmt[21 + 1];       // �ѳɽ����
    char          chIsWithdraw;               // ������־
    char          chIsWithdrawn;              // �ѳ�����־
    char          chOptUndlCls;               // ���֤ȯ���
    char          szOptUndlCode[8 + 1];       // ���֤ȯ����
    char          szOptUndlName[16 + 1];      // ���֤ȯ����
    LONGLONG      llUndlFrzQty;               // ���ȯί�ж�������
    LONGLONG      llUndlUfzQty;               // ���ȯί�нⶳ����
    LONGLONG      llUndlWthQty;               // ���ȯ�ѳ�������
    char          szOfferRetMsg[64 + 1];      // �걨������Ϣ
    short         iStrategySn;                // ���Ա��
    int           iOrderSn;                   // ί�����
    char          szRawOrderId[10 + 1];       // ԭ��ͬ���
    char          szMarginPreFrz[21 + 1];     // Ԥռ�ñ�֤�� ����ί��ʱ��дԤ����ı�֤�����������0��
    char          szMarginFrz[21 + 1];        // ռ�ñ�֤�� �����ɽ�ʱ��дʵ�ʶ���ı�֤�����������0��
    char          szMarginPreUfz[21 + 1];     // Ԥ�ⶳ��֤�� ��ƽί��ʱ��дԤ�ⶳ�ı�֤�����������0��
    char          szMarginUfz[21 + 1];        // �ⶳ��֤�� ��ƽ�ɽ�ʱ��дʵ�ʽⶳ�ı�֤�����������0��
};

//-------------------------------��Ȩ���ճɽ���ѯ------------------------------------
struct STReqOptQryFill
{
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szStkbd[2 + 1];             // ���װ�� �ֵ�[STKBD]
    char          szTrdacct[10 + 1];          // �����˻�
    char          szOptNum[32 + 1];           // ��Լ����
    char          szOptUndlCode[8 + 1];       // ���֤ȯ����
    char          szCombStraCode[16 + 1];     // ��ϲ��Դ���
    char          szOrderId[10 + 1];          // ��ͬ���
    int           iOrderBsn;                  // ί������
    char          chQueryFlag;                // ��ѯ���� 0:���ȡ���� 1:��ǰȡ���� ����ȫ������
    char          szQryPos[32 + 1];           // ��λ��
    int           iQryNum;                    // ��ѯ����
};

struct STRspOptQryFill
{
    char          szQryPos[32 + 1];           // ��λ��
    int           iTrdDate;                   // ��������
    char          szMatchedTime[8 + 1];       // �ɽ�ʱ��
    int           iOrderDate;                 // ί������
    int           iOrderSn;                   // ί�����
    int           iOrderBsn;                  // ί������
    char          szOrderId[10 + 1];          // ��ͬ���
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szStkbd[2 + 1];             // ���װ��
    char          szStkpbu[8 + 1];            // ���׵�Ԫ
    char          szTrdacct[10 + 1];          // ֤ȯ�˻�
    char          szSubacctCode[8 + 1];       // ֤ȯ�˻��ӱ���
    char          szOptTrdacct[18 + 1];       // ��Ȩ��Լ�˻�
    int           iStkBiz;                    // ֤ȯҵ��
    int           iStkBizAction;              // ֤ȯҵ����Ϊ
    char          szOwnerType[3 + 1];         // ������������
    char          szOptNum[16 + 1];           // ��Լ����
    char          szOptCode[32 + 1];          // ��Լ����
    char          szOptName[32 + 1];          // ��Լ���
    char          szCombNum[16 + 1];          // ��ϱ���
    char          szCombStraCode[16 + 1];     // ��ϲ��Դ���
    char          szLeg1Num[16 + 1];          // �ɷ�һ��Լ����
    char          szLeg2Num[16 + 1];          // �ɷֶ���Լ����
    char          szLeg3Num[16 + 1];          // �ɷ�����Լ����
    char          szLeg4Num[16 + 1];          // �ɷ��ĺ�Լ����
    char          chCurrency;                 // ���Ҵ���
    char          chOptUndlCls;               // ���֤ȯ���
    char          szOptUndlCode[8 + 1];       // ���֤ȯ����
    char          szOptUndlName[16 + 1];      // ���֤ȯ����
    char          szOrderPrice[11 + 1];       // ί�м۸�
    LONGLONG      llOrderQty;                 // ί������
    char          szOrderAmt[21 + 1];         // ί�н��
    char          szOrderFrzAmt[21 + 1];      // ί�ж�����
    char          chIsWithdraw;               // ������־
    char          chMatchedType;              // �ɽ�����
    char          szMatchedSn[32 + 1];        // �ɽ����
    char          szMatchedPrice[11 + 1];     // �ɽ��۸�
    LONGLONG      llMatchedQty;               // �ѳɽ�����
    char          szMatchedAmt[21 + 1];       // �ѳɽ����
    short         iStrategySn;                // ���Ա��
    char          szMarginPreFrz[21 + 1];     // Ԥռ�ñ�֤�� ����ί��ʱ��дԤ����ı�֤�����������0��
    char          szMarginFrz[21 + 1];        // ռ�ñ�֤�� �����ɽ�ʱ��дʵ�ʶ���ı�֤�����������0��
    char          szMarginPreUfz[21 + 1];     // Ԥ�ⶳ��֤�� ��ƽί��ʱ��дԤ�ⶳ�ı�֤�����������0��
    char          szMarginUfz[21 + 1];        // �ⶳ��֤�� ��ƽ�ɽ�ʱ��дʵ�ʽⶳ�ı�֤�����������0��
    char          szMatchedFee[21 + 1];       // �ɽ�����
};


//-------------------------------��Ȩ�ɳ�ί�в�ѯ------------------------------------
struct STReqOptQryWithdrawableOrder
{
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szStkbd[2 + 1];             // ���װ��
    char          szTrdacct[10 + 1];          // �����˻�
    char          szOptNum[32 + 1];           // ��Լ����
    char          szOptUndlCode[8 + 1];       // ���֤ȯ����
    char          szOrderId[10 + 1];          // ��ͬ���
    char          chQueryFlag;                // ��ѯ���� 0:���ȡ���� 1:��ǰȡ���� ����ȫ������
    char          szQryPos[32 + 1];           // ��λ��
    int           iQryNum;                    // ��ѯ����
};

struct STRspOptQryWithdrawableOrder
{
    char          szQryPos[32 + 1];           // ��λ��
    int           iTrdDate;                   // ��������
    int           iOrderDate;                 // ί������
    char          szOrderTime[25 + 1];        // ί��ʱ��
    int           iOrderBsn;                  // ί������
    char          szOrderId[10 + 1];          // ��ͬ���
    char          chOrderStatus;              // ί��״̬
    char          chOrderValidFlag;           // ί����Ч��־
    LONGLONG      llCuacctCode;               // �ʲ��˻�
    char          szStkbd[2 + 1];             // ���װ��
    char          szTrdacct[10 + 1];          // ֤ȯ�˻�
    char          szSubacctCode[8 + 1];       // ֤ȯ�˻��ӱ���
    char          szOptTrdacct[18 + 1];       // ��Ȩ��Լ�˻�
    int           iStkBiz;                    // ֤ȯҵ��
    int           iStkBizAction;              // ֤ȯҵ����Ϊ
    char          szOwnerType[3 + 1];         // ������������
    char          szOptNum[16 + 1];           // ��Լ����
    char          szOptCode[32 + 1];          // ��Լ����
    char          szOptName[32 + 1];          // ��Լ���
    char          chCurrency;                 // ���Ҵ���
    char          szOrderPrice[11 + 1];       // ί�м۸�
    LONGLONG      llOrderQty;                 // ί������
    char          szOrderAmt[21 + 1];         // ί�н��
    char          szOrderFrzAmt[21 + 1];      // ί�ж�����
    char          szOrderUfzAmt[21 + 1];      // ί�нⶳ���
    LONGLONG      llOfferQty;                 // �걨����
    int           iOfferStime;                // �걨ʱ��
    LONGLONG      llWithdrawnQty;             // �ѳ�������
    LONGLONG      llMatchedQty;               // �ѳɽ�����
    char          szMatchedAmt[21 + 1];       // �ѳɽ����
    char          chIsWithdraw;               // ������־
    char          chIsWithdrawn;              // �ѳ�����־
    char          chOptUndlCls;               // ���֤ȯ���
    char          szOptUndlCode[8 + 1];       // ���֤ȯ����
    char          szOptUndlName[16 + 1];      // ���֤ȯ����
    LONGLONG      llUndlFrzQty;               // ���ȯί�ж�������
    LONGLONG      llUndlUfzQty;               // ���ȯί�нⶳ����
    LONGLONG      llUndlWthQty;               // ���ȯ�ѳ�������
    char          szMarginPreFrz[21 + 1];     // Ԥռ�ñ�֤�� ����ί��ʱ��дԤ����ı�֤�����������0��
    char          szMarginFrz[21 + 1];        // ռ�ñ�֤�� �����ɽ�ʱ��дʵ�ʶ���ı�֤�����������0��
    char          szMarginPreUfz[21 + 1];     // Ԥ�ⶳ��֤�� ��ƽί��ʱ��дԤ�ⶳ�ı�֤�����������0��
    char          szMarginUfz[21 + 1];        // �ⶳ��֤�� ��ƽ�ɽ�ʱ��дʵ�ʽⶳ�ı�֤�����������0��
};
#endif  //__ZS_STK_TRADE_API_STRUCT_H__