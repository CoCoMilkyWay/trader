#if !defined(__STK_TRADE_API_H__)
#define __STK_TRADE_API_H__

#include "baseApi.h"
#include "tradeApiStruct.h"


class ZSAPI CTradeCallback : virtual public CBaseCallback
{
public:
    // ί����Ӧ
    virtual int OnOrder(STFirstSet *p_pFirstSet, STRspOrder *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ������Ӧ
    virtual int OnCancelOrder(STFirstSet *p_pFirstSet, STRspCancelOrder *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ���ɽ�����������Ӧ
    virtual int OnMaxTradeQty(STFirstSet *p_pFirstSet, STRspMaxTradeQty *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �ֲֲ�ѯ��Ӧ
    virtual int OnQryHolding(STFirstSet *p_pFirstSet, STRspQryHolding *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �ɽ���ѯ��Ӧ
    virtual int OnQryFill(STFirstSet *p_pFirstSet, STRspQryFill *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �ʽ��ѯ��Ӧ
    virtual int OnQryMoney(STFirstSet *p_pFirstSet, STRspQryMoney *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ί�в�ѯ��Ӧ
    virtual int OnQryOrder(STFirstSet *p_pFirstSet, STRspQryOrder *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �ɶ��˻���ѯ��Ӧ
    virtual int OnQryHolder(STFirstSet *p_pFirstSet, STRspQryHolder *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �ɳ���ί�в�ѯ��Ӧ
    virtual int OnQryWithdrawableOrder(STFirstSet *p_pFirstSet, STRspQryWithdrawableOrder *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ���Խ���ע����Ӧ
    virtual int OnAcctRegister(STFirstSet *p_pFirstSet, STRspAcctRegister *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �����˻���¼��Ӧ
    virtual int OnTradeLogin(STFirstSet *p_pFirstSet, STRspTradeLogin *p_pRsp, LONGLONG  p_llReqId, int p_iNum) {return 0;}

    // ϵͳ�û���¼��Ӧ
    virtual int OnLogin(STFirstSet *p_pFirstSet, STRspLogin *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ȷ�ϻر�
    virtual int OnRtnOrderConfirm(STRtnOrderConfirm *p_pRtnOrderConfirm) { return 0; }

    // �ɽ��ر�
    virtual int OnRtnOrderFill(STRtnOrderFill *p_pRtnOrderFill) { return 0; }

public:
    // ������ȯί���µ���Ӧ
    virtual int OnOrderCredit(STFirstSet *p_pFirstSet, STRspOrderCredit *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ������ȯֱ�ӻ�����Ӧ
    virtual int OnRepay(STFirstSet *p_pFirstSet, STRspRepay *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ������ȯ�ֲֲ�ѯ��Ӧ
    virtual int OnQryHoldingCredit(STFirstSet *p_pFirstSet, STRspQryHoldingCredit *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ������ȯ���ȯ��Ϣ��ѯ��Ӧ
    virtual int OnQryUndlStkInfo(STFirstSet *p_pFirstSet, STRspQryUndlStkInfo *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ������ȯ����֤ȯ��Ϣ��ѯ��Ӧ
    virtual int OnQryColStkInfo(STFirstSet *p_pFirstSet, STRspQryColStkInfo *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ������ȯ��Լ��ѯ��Ӧ
    virtual int OnQryContract(STFirstSet *p_pFirstSet, STRspQryContract *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ���ÿͻ��ʲ���ծ��ѯ��Ӧ
    virtual int OnQryCustDebts(STFirstSet *p_pFirstSet, STRspQryCustDebts *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

public:
    // ֤ȯ��ֵ��Ȳ�ѯ��Ӧ
    virtual int OnQryMktQuota(STFirstSet *p_pFirstSet, STRspQryMktQuota *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �����¹���Ϣ��ѯ��Ӧ
    virtual int OnQryIpoInfo(STFirstSet *p_pFirstSet, STRspQryIpoInfo *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ת������ҵ����Ϣ��ѯ��Ӧ
    virtual int OnQryBankInfo(STFirstSet *p_pFirstSet, STRspQryBankInfo *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ��֤ת����Ӧ
    virtual int OnBankStkTrans(STFirstSet *p_pFirstSet, STRspBankStkTrans *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �����˻�����ѯ��Ӧ
    virtual int OnQryBankBalance(STFirstSet *p_pFirstSet, STRspQryBankBalance *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ��֤ת�ʲ�ѯ��Ӧ
    virtual int OnQryBankStkTransInfo(STFirstSet *p_pFirstSet, STRspQryBankStkTransInfo *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �޸Ľ���������Ӧ
    virtual int OnModifyTradePwd(STFirstSet *p_pFirstSet, STRspModifyTradePwd *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �޸��ʽ�������Ӧ
    virtual int OnModifyFundPwd(STFirstSet *p_pFirstSet, STRspModifyFundPwd *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ���н��׹�̨�ʽ��ѯ��Ӧ
    virtual int OnQryRpcFund(STFirstSet *p_pFirstSet, STRspQryRpcFund *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // �ʽ𻮲�������Ӧ
    virtual int OnFundTransfer(STFirstSet *p_pFirstSet, STRspFundTransfer *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ��֤ת�������˺Ų�ѯ��Ӧ
    virtual int OnQryBankAcct(STFirstSet *p_pFirstSet, STRspQryBankAcct *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

public:
    // ��Ȩί���걨��Ӧ
    virtual int OnOrderOpt(STFirstSet *p_pFirstSet, STRspOrderOpt *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ��Ȩί�г�����Ӧ
    virtual int OnOptCancelOrder(STFirstSet *p_pFirstSet, STRspOptCancelOrder *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ��Ȩ�����ʽ��ѯ��Ӧ
    virtual int OnOptQryMoney(STFirstSet *p_pFirstSet, STRspOptQryMoney *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ��Ȩ�ֲֲ�ѯ��Ӧ
    virtual int OnOptQryHolding(STFirstSet *p_pFirstSet, STRspOptQryHolding *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ��Ȩ����ί�в�ѯ��Ӧ
    virtual int OnOptQryOrder(STFirstSet *p_pFirstSet, STRspOptQryOrder *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ��Ȩ���ճɽ���ѯ��Ӧ
    virtual int OnOptQryFill(STFirstSet *p_pFirstSet, STRspOptQryFill *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

    // ��Ȩ�ɳ�ί�в�ѯ��Ӧ
    virtual int OnOptQryWithdrawableOrder(STFirstSet *p_pFirstSet, STRspOptQryWithdrawableOrder *p_pRsp, LONGLONG  p_llReqId, int p_iNum) { return 0; }

private:

};

class ZSAPI CTradeApi : public CBaseApi
{
public:
    CTradeApi(void);
    virtual ~CTradeApi(void);

    // ί������
    int Order(STReqOrder *p_pReq, LONGLONG p_llReqId);

    // ��������
    int CancelOrder(STReqCancelOrder *p_pReq, LONGLONG p_llReqId);

    // ���ɽ�������������
    int MaxTradeQty(STReqMaxTradeQty *p_pReq, LONGLONG p_llReqId);

    // �ֲֲ�ѯ����
    int QryHolding(STReqQryHolding *p_pReq, LONGLONG p_llReqId);

    // �ɽ���ѯ����
    int QryFill(STReqQryFill *p_pReq, LONGLONG p_llReqId);

    // �ʽ��ѯ����
    int QryMoney(STReqQryMoney *p_pReq, LONGLONG p_llReqId);

    // ί�в�ѯ����
    int QryOrder(STReqQryOrder *p_pReq, LONGLONG p_llReqId);

    // �ɶ��˻���ѯ����
    int QryHolder(STReqQryHolder *p_pReq, LONGLONG p_llReqId);

    // �ɳ���ί�в�ѯ����
    int QryWithdrawableOrder(STReqQryWithdrawableOrder *p_pReq, LONGLONG p_llReqId);

    // ���Խ���ע������
    int AcctRegister(STReqAcctRegister *p_pReq, LONGLONG p_llReqId);

    // �����˻���¼����
    int TradeLogin(STReqTradeLogin *p_pReq, LONGLONG p_llReqId);

    // ϵͳ�û���¼����
    int Login(STReqLogin *p_pReq, LONGLONG p_llReqId);

public:
    // ������ȯί���µ�����
    int OrderCredit(STReqOrderCredit *p_pReq, LONGLONG p_llReqId);

    // ������ȯֱ�ӻ�������
    int Repay(STReqRepay *p_pReq, LONGLONG p_llReqId);

    // ������ȯ�ֲֲ�ѯ����
    int QryHoldingCredit(STReqQryHoldingCredit *p_pReq, LONGLONG p_llReqId);

    // ������ȯ���ȯ��Ϣ��ѯ����
    int QryUndlStkInfo(STReqQryUndlStkInfo *p_pReq, LONGLONG p_llReqId);

    // ������ȯ����֤ȯ��Ϣ��ѯ����
    int QryColStkInfo(STReqQryColStkInfo *p_pReq, LONGLONG p_llReqId);

    // ������ȯ��Լ��ѯ����
    int QryContract(STReqQryContract *p_pReq, LONGLONG p_llReqId);

    // ���ÿͻ��ʲ���ծ��ѯ����
    int QryCustDebts(STReqQryCustDebts *p_pReq, LONGLONG p_llReqId);

public:
    // ֤ȯ��ֵ��Ȳ�ѯ����
    int QryMktQuota(STReqQryMktQuota *p_pReq, LONGLONG p_llReqId);

    // �����¹���Ϣ��ѯ����
    int QryIpoInfo(STReqQryIpoInfo *p_pReq, LONGLONG p_llReqId);

    // ת������ҵ����Ϣ��ѯ����
    int QryBankInfo(STReqQryBankInfo *p_pReq, LONGLONG p_llReqId);

    // ��֤ת������
    int BankStkTrans(STReqBankStkTrans *p_pReq, LONGLONG p_llReqId);

    // �����˻�����ѯ����
    int QryBankBalance(STReqQryBankBalance *p_pReq, LONGLONG p_llReqId);

    // ��֤ת�ʲ�ѯ����
    int QryBankStkTransInfo(STReqQryBankStkTransInfo *p_pReq, LONGLONG p_llReqId);

    // �޸Ľ�����������
    int ModifyTradePwd(STReqModifyTradePwd *p_pReq, LONGLONG p_llReqId);

    // �޸��ʽ���������
    int ModifyFundPwd(STReqModifyFundPwd *p_pReq, LONGLONG p_llReqId);

    // ���н��׹�̨�ʽ��ѯ����
    int QryRpcFund(STReqQryRpcFund *p_pReq, LONGLONG p_llReqId);

    // �ʽ𻮲�����
    int FundTransfer(STReqFundTransfer *p_pReq, LONGLONG p_llReqId);

    // ��֤ת�������˺Ų�ѯ����
    int QryBankAcct(STReqQryBankAcct *p_pReq, LONGLONG p_llReqId);

public:
    // ��Ȩί���걨����
    int OrderOpt(STReqOrderOpt *p_pReq, LONGLONG p_llReqId);

    // ��Ȩί�г�������
    int OptCancelOrder(STReqOptCancelOrder *p_pReq, LONGLONG p_llReqId);

    // ��Ȩ�����ʽ��ѯ����
    int OptQryMoney(STReqOptQryMoney *p_pReq, LONGLONG p_llReqId);

    // ��Ȩ�ֲֲ�ѯ����
    int OptQryHolding(STReqOptQryHolding *p_pReq, LONGLONG p_llReqId);

    // ��Ȩ����ί�в�ѯ����
    int OptQryOrder(STReqOptQryOrder *p_pReq, LONGLONG p_llReqId);

    // ��Ȩ���ճɽ���ѯ����
    int OptQryFill(STReqOptQryFill *p_pReq, LONGLONG p_llReqId);

    // ��Ȩ�ɳ�ί�в�ѯ����
    int OptQryWithdrawableOrder(STReqOptQryWithdrawableOrder *p_pReq, LONGLONG p_llReqId);

public:
    virtual void OnReqCallback(const int p_iMsgId, const void *p_pszDataBuff, int p_iDataLen, LONGLONG p_llReqId);
    void OnSubCallback(const char *p_pszMsg, const void *p_pszDataBuff, int p_iDataLen);

private:
    void OnAcctRegister(STFirstSet *p_pFirstSet, STRspAcctRegister *p_pRsp);
    void OnLogin(STFirstSet *p_pFirstSet, STRspLogin *p_pRsp);
    void OnTradeLogin(STFirstSet *p_pFirstSet, STRspTradeLogin *p_pRsp);
};


#endif  //__STK_TRADE_API_H__