#pragma once

#ifndef SWESOME_CHAR	
#define SWESOME_CHAR
typedef char    char_8[8];
typedef char	char_10[10];
typedef char	char_20[20];
typedef char	char_31[31];
typedef char	char_32[32];
typedef char	char_64[64];
typedef char	char_128[128];
typedef char	char_256[256];
#endif

#pragma pack(1)

//�г�����(�˶����������ʹ��)
#define		ACLT_MARKET_UNKNOW			0
#define		ACLT_MARKET_SSE				1
#define		ACLT_MARKET_SZSE			2
#define		ACLT_MARKET_CFFEX			3
#define		ACLT_MARKET_CNF				4
typedef		char	AClt_Market;

//.................................................................................................................................................................................................................................
//�г�ʱ��
typedef struct
{
	int								Date;						//���ڣ���ʽΪYYYYMMDD
	int								Time;						//ʱ��, ��ʽΪHHMMSS
}tagAClt_MarketField;

//һ��
typedef struct
{
	double							Price;						//ί�м۸�
	unsigned __int64				Volume;						//ί����[��]
}tagAClt_BuySell;

//����
typedef struct
{
	AClt_Market						Exchange;
	char_31							Code;
	double		 					Open;						//���̼�
	double		 					High;						//��߼�
	double		 					Low;						//��ͼ�
	double		 					Now;						//����

	unsigned __int64 				Volume;						//�ɽ���
	double 							Amount;						//�ɽ����(Ԫ)

	unsigned __int64 				Position;					//�ֲ���
	double		 					SettlePrice;				//�����

	tagAClt_BuySell					Buy[5];						//��5��
	tagAClt_BuySell					Sell[5];					//��5��
	char_8 							TradingCode;				//����״̬(����)[���ĵ�]
}tagAClt_QuoteField;

//.................................................................................................................................................................................................................................
typedef struct
{
	AClt_Market						Exchange;
	char_31							Code;
}tagAClt_Instrument;

#define		AClt_INSTRTYPE_STOCK		1
#define		AClt_INSTRTYPE_OPTION		2
#define		AClt_INSTRTYPE_FUTURE		3
typedef     char		ACLT_INSTRUMENT_TYPE;

//��Ʊ������������ݽṹ
typedef struct
{
	tagAClt_Instrument				Instrument;					//�г�+����
	char_32							Name;						//����or���(GBK����)[ע��:���ڽ��������ܻ���λ,��λ������ƣ�ͨ�������ӿڻ�ȡ] 
	double							PreClose;					//���ռ�
	double							UpperLimit;					//��ͣ��
	double							LowerLimit;					//��ͣ��

	unsigned char					SubType;					//��Ʊor ETFFund
	char_10							ProductID;					//������ƷID

	unsigned int					LotSize;					//�ֱ���
	unsigned int					ContractMulti;				//(��Լ����*��Լ��λ)
	double							PriceTick;					//�۸�䶯��λ
	unsigned char					ShowDot;					//��ʾС��λ��
	bool							IsTrading;					//�Ƿ���
}tagAClt_StockBaseData;

//.................................................................................................................................................................................................................................
//��Ȩ�������ݽṹ
typedef struct
{
	tagAClt_Instrument				Instrument;					//�г�+����(�˴����ǿ����µ��Ĵ���)
	char_32							Name;					//����or ���(GBK)		//�н������ [ע��:���ڽ��������ܻ���λ,��λ������ƣ�ͨ�������ӿڻ�ȡ] 
	char_20							ContractID;				//������Ȩ�ĺ�Լ���룬���������µ�  //�н����

	double							PreClose;					//���ռ�
	double							PreSettle;					//����
	unsigned __int64				PrePosition;				//��ֲ�
	double							UpperLimit;					//��ͣ��
	double							LowerLimit;					//��ͣ��

	char							OptKind;					//��Ȩ����('C' = call 'P'= put)
	char							ExecKind;					//��Ȩ����('A' = ��ʽ 'E' = ŷʽ)

	char_10							ProductID;					//������ƷID
	tagAClt_Instrument				UnderlyingCode;				//����г�+����

	unsigned int					LotSize;					//�ֱ���
	unsigned int					ContractMulti;				//(��Լ����*��Լ��λ)
	double							PriceTick;					//�۸�䶯��λ

	double							ExecPrice;					//��Ȩ�۸�
	int								LastTradeDay;				//�������
	int								EndDay;						//������

	unsigned char					ShowDot;					//��ʾС��λ��
	double							reserved;				//reserved
	bool							IsTrading;					//�Ƿ���
}tagAClt_OptionBaseData;

//�ڻ��������ݽṹ
typedef struct
{
	tagAClt_Instrument				Instrument;					//�г�+����
	char_32							Name;					//����or ���(GBK) [ע��:���ڽ��������ܻ���λ,��λ������ƣ�ͨ�������ӿڻ�ȡ] 
	double							PreClose;					//���ռ�
	double							PreSettle;					//����
	unsigned __int64				PrePosition;				//��ֲ�
	double							UpperLimit;					//��ͣ��
	double							LowerLimit;					//��ͣ��

	char_10							ProductID;				//������ƷID
	tagAClt_Instrument				UnderlyingCode;				//����г�+����

	unsigned int					LotSize;					//�ֱ���
	unsigned int					ContractMulti;				//(��Լ����*��Լ��λ)
	double							PriceTick;					//�۸�䶯��λ

	int								LastTradeDay;				//�������
	int								EndDay;						//������
	unsigned char					ShowDot;					//��ʾС��λ��
	bool							IsTrading;					//�Ƿ���
}tagAClt_FutureBaseData;


//������������һ
typedef	struct
{
	ACLT_INSTRUMENT_TYPE			InstrType;					//��Ʒ����
	union
	{
		tagAClt_StockBaseData			StockBase;					//��Ʊ��
		tagAClt_OptionBaseData			OptionBase;					//��Ȩ��
		tagAClt_FutureBaseData			FutureBase;					//�ڻ���
	};
}tagAClt_CommBaseData;


//��������(1.14+)
typedef struct  
{
	AClt_Market						Exchange;
	char_31							Code;
	char_64							Name;
	
	char							Reserved[1024];
}tagAClt_SupplementData;


#pragma pack()
