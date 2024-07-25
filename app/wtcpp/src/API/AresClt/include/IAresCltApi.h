#pragma once
#include "IAresCltStruct.h"

class IAresCltSpi
{
public:
	//֪ͨ���г����ں�ʱ��
	virtual		void			OnMarketTime(AClt_Market cMarket, tagAClt_MarketField*) = 0;
	//֪ͨ���������
	virtual		void			OnMarketData(AClt_Market cMarket, tagAClt_QuoteField*) = 0;

};

class IAresExchange
{
public:
	// ȡ��������Ʒ������
	virtual		int				GetCommodityCount() = 0;
	// ȡ��������Ʒ���
	virtual		int				GetCommodityData(tagAClt_Instrument* pArr, int nCount) = 0;
	// ȡĳһ��Ʒ��������
	virtual		int				GetOneStaticData(tagAClt_Instrument* pInstrument, tagAClt_CommBaseData* pData) = 0;
	// ȡ��������Ʒ��������
	virtual		int				GetStaticData(tagAClt_CommBaseData* pArr, int nCount) = 0;

	// ȡĳһ��Ʒ�Ĳ�������(1.14+)
	virtual		int				GetOneSupplementData(tagAClt_Instrument* pInstrument, tagAClt_SupplementData* pData) = 0;
	// ȡ��������Ʒ��������(1.14+)
	virtual		int				GetSupplementData(tagAClt_SupplementData* pArr, int nCount) = 0;

};

class IAresCltApi
{
public:
	//ע������֪ͨSpi
	virtual		void			RegisterSpi(IAresCltSpi*) = 0;
	//��������
	virtual		int				StartWork() = 0;
	//ֹͣ����
	virtual		void			EndWork() = 0;

public:
	//��ȡ�г��б�
	virtual	 IAresExchange*		GetExchPtr(AClt_Market cMarket) = 0;

};


























