#pragma once
#pragma pack(push,1)
namespace MarItpdk
{


//�ɶ���
typedef struct
{
   char     AccountId[16];    //�ͻ���
   char     Market[4];        //������
   char     SecuAccount[12];  //�ɶ���
   char     HolderName[16];   //�ɶ�����
   char     FundAccount[16];  //�ʽ��ʺ�
   char     OrgCode[8];       //��������--��������
   char     MoneyType[4];     //����
   char     TradeAccess[16];  //����Ȩ��
   char     MarketingUnit[8]; //���׵�Ԫ
   int64    HolderType;       //�ɶ����
} ITPDK_XYGDH;


//����Ȩ��
typedef struct
{
   char     AccountId[16];    //�ͻ���
   char     Market[4];        //������
   char     SecuAccount[12];  //�ɶ���
   int      StarQty;          //�ƴ�������
   int      EquityQty;        //֤ȯȨ������
   int      SettleMentDate;   //��������
} ITPDK_XYPSQY;



//����֤ȯ�����ѯ
typedef struct
{
    char     Market[4];           //������
    char     MoneyType[4];        //����
    char     StockCode[12];       //֤ȯ����
    char     StockName[16];       //֤ȯ����
    char     StockType[4];        //֤ȯ���
    double   PriceTick;           //���׼�λ
    double   TradeUnit;           //���׵�λ
    int64    MaxMarketTradeAmt;        //�м�ί������
    int64    MinMarketTradeAmt;        //�м�ί������
    int64      MaxTradeAmt;         //ί������
    int64      MinTradeAmt;         //ί������
    double   LastClosePrice;      //������
    double   HighLimitPrice;      //��߱���--��ͣ��
    double   LowLimitPrice;       //��ͱ���--��ͣ��
    int64      NetPriceFlag;        //��ծ���۱�־
    double   InterestPrice;       //��Ϣ����
    int64     BrowIndex;           //��ҳ��ѯ��λ��
   
}ITPDK_XYZQDM;


//�ʽ���Ϣ
typedef struct
{
   char     AccountId[16];             //�ͻ���
   char     FundAccount[16];           //�ʽ��˺�
   char     MoneyType[4];              //����
   char     OrgCode[8];                //��������
   int      MasterFlag;                //���ʻ���־
   int      AccountType;               //�ʻ����
   double   LastBalance;               //�������
   double   CurrentBalance;            //�˻����
   double   FrozenBalance;             //�����ʽ�
   double   T2_FrozenBalance;          //T+2������
   double   FundAvl;                   //�����ʽ�
   double   T2_FundAvl;                //T+2�����ʽ�
   double   TotalAsset;                //���ʲ�
   double   MarketValue;               //������ֵ
   double   DebtAmt;                   //��ծ���
   double   CreditQuota;               //���ö��
   double   CreditQuotaAvl;            //�������ö��
   double   UncomeBalance;             //δ�����ʽ�
   double   CashBalance;               //�ֽ����
   double   CashAsset;                 //�ֽ��ʲ�
   double   OtherAsset;                //�����ʲ�
   double   FetchBalance;              //��ȡ�ʽ�
   double   DateProfit;                //����ӯ��
   double   UnclearProfit;             //����ӯ��
   double   DiluteUnclearProfit;       //̯������ӯ��
   double   UpdateTime;                //����ʱ��
   double 	SettleBalance;				// ʵʱ�����ʽ�
   double   ContractPosiValue;				//��Լ�ֲ���ֵ(��չ��ѯ)
   double   LastPositionValue;			//���ճֲ���ֵ(��չ��ѯ)
} ITPDK_XYZJZH;


//֤ȯ�ֲ�
typedef struct
{
   char     AccountId[16];             //�ͻ���
   char     Market[4];                 //������
   char     StockCode[12];             //֤ȯ����
   char     SecuAccount[12];           //�ɶ���
   char     FundAccount[16];           //�ʽ��˺�
   int64      AccountType;               //�˻����
   char     MoneyType[4];              //����
   char     StockName[16];             //֤ȯ����
   int64      CurrentQty;                //��ֲ���
   int64      QtyAvl;                    //����������
   double   LastPrice;                 //���¼�
   double   MarketValue;               //������ֵ
   double   DateProfit;                //����ӯ��
   double   CostPrice;                 //�ֲ־���
   double   UnclearProfit;             //����ӯ��
   double   DividendAmt;               //�������
   double   RealizeProfit;             //ʵ��ӯ��
   int64   PreQty;                    //��ֲ���
   int64   FrozenQty;                 //��������
   int64    OrderFrozenQty;             //ί�ж�������
   int64   UncomeQty;                 //δ��������
   double   CostBalance;               //�ֲֳɱ�
   double   DiluteCostPrice;           //̯���ɱ���
   double   KeepCostPrice;             //������
   double   AvgBuyPrice;               //�������
   int64   AllotmentQty;              //�������
   int64   SubscribeQty;              //�깺����
   char     OpenDate[12];              //��������
   double   InterestPrice;             //��Ϣ����
   double   Dilutekeep_CostPrice;      //̯��������
   double   DiluteUnclearProfit;       //̯������ӯ��
   int64      TradeUnit;                 //���׵�λ
   int64      SubscribableQty;           //���깺����
   int64      RedeemableQty;             //���������
   int64      RealSubsQty;               //�깺�ɽ�����
   int64      RealRedeQty;               //��سɽ�����
   int64   TotalSellQty;              //�ۼ���������
   int64   TotalBuyQty;               //�ۼ���������
   double   TotalSellAmt;              //�������
   double   TotalBuyAmt;               //������
   double   AllotmentAmt;              //��ɽ��
   int64      RealBuyQty;                //��������ɽ�����
   int64      RealSellQty;               //���������ɽ�����
   double   RealBuyBalance;            //��������ɽ����
   double   RealSellBalance;           //���������ɽ����
   int64      BrowIndex;                 //��ҳ��ѯ��λ��
} ITPDK_XYZQGL;
//����ί��
typedef struct
{
   char     AccountId[16];       //�ͻ���
   int      OrderId;             //ί�к�
   int      CancelOrderId;       //����ί�к�
   char     SBWTH[17];           //�걨ί�к�
   char     Market[4];           //������
   char     StockCode[12];       //֤ȯ����
   char     StockName[16];       //֤ȯ����
   int      EntrustType;         //�������
   double   OrderPrice;          //ί�м۸�
   int      OrderQty;            //ί������
   double   MatchPrice;          //�ɽ��۸�
   int   MatchQty;            //�ɽ�����
   int      WithdrawQty;         //��������
   char     SecuAccount[12];     //�ɶ���
   int      BatchNo;             //ί�����κ�
   int      EntrustDate;         //ί������
   int      SerialNo;            //��ˮ��
   int      OrderType;           //��������
   int      OrderType_HK;        //�۹ɶ�������
   double   StopPrice;           //ֹ���޼�
   int      OrderStatus;         //�걨���
   char     EntrustNode[48];     //����վ��
   char     EntrustTime[13];     //ί��ʱ��
   char     ReportTime[13];      //�걨ʱ��
   char     MatchTime[13];       //�ɽ�ʱ��
   char     KfsOrderNum[24];       //�����̱������
   char     WithdrawFlag[4];     //������־
   char     ResultInfo[128];     //���˵��
   double   MatchAmt;            //�ɽ����
   double   FrozenBalance;       //�����ʽ�
   double   BailBalance;         //���ᱣ֤��
   double   HandingFee;          //�����̷ּ�
   int      BrowIndex;           //��ҳ��ѯ��λ��
} ITPDK_XYDRWT;

//�ֱʳɽ�
typedef struct
{
   char     AccountId[16];       //�ͻ���
   int64      ReportSerialNo;      //�ر����
   char     Market[4];           //������
   char     SecuAccount[12];     //�ɶ���
   char     MatchSerialNo[32];   //�ɽ����
   int64      OrderId;             //ί�к�
   int64      EntrustType;         //�������
   char     MatchTime[13];       //�ɽ�ʱ��
   char     StockCode[12];       //֤ȯ����
   char     StockName[16];       //֤ȯ����
   char     KfsOrderNum[24];       //�����̱������
   int64   MatchQty;            //�ɽ�����
   double   MatchPrice;          //�ɽ��۸�
   double   MatchAmt;            //�ɽ����
   char     MoneyType[4];        //����
   double   clearBalance;        //������
   int64      BatchNo;             //ί�����κ�
   int64      EntrustDate;         //ί������
   int64      BrowIndex;           //��ʼ��¼����ֵ
   char     WithdrawFlag[4];     //������־
} ITPDK_XYSSCJ;


//�ʸ�֤ȯ
typedef struct
{
    char     Market[4];           //������
    char     StockCode[12];       //֤ȯ����
    char     StockName[16];       //֤ȯ����
    double   ConversionRate;      //������
    double   MarginBailRate;      //���ʱ�֤�����
    double   ShortBailRate;       //��ȯ��֤�����
    int      StockProperty;       //֤ȯ���� 1	���ʱ��, 2	��ȯ���, 4	������, 7 ������ȯ�������
    int      TradeStatus;         //����״̬  0-������1 ��ֹ���� 2 ��ֹ��ȯ 3 ��ֹ������ȯ
    int      CollateralStatus;    //����״̬ 0-���� 1-��ͣ 2-���� 3-ͬ��׼����
    int      Type;                //���ͣ�0-��˾��1 ����
    int64    BrowIndex;           //��ҳ��ѯ��λ��
} ITPDK_ZGZQ;

//�ͻ�����ȯ��Ϣ
typedef struct
{
    char        AccountId[16];       //�ͻ���
    int32       nTCXZ;      //ͷ�����ʣ�1��ͨͷ�� 2ר��ͷ�磩
    char        szTCBH[21];  //ͷ����
    char        sJYS[3];    //������
    char        sZQDM[11];  //֤ȯ����
    char        sZQMC[21];  //֤ȯ����
    int64       nZQSL;      //֤ȯ����
    int64       nDJSL;      //��������
    int64       nMCWTSL;    //����ί������
    int64       nDRCHSL;    //���ճ�������
    int64       nRCSL;      //���ڳ�����
    int64       nYYSL;      //ԤԼ����
    int64       nDRCJSL;    //���ճɽ�����
    int64       nKYSL;      //��������
    int64       nHCCHSL;    //������������
    float64     dRQBL;      //��ȯ����
    float64     dZSL;       //������
    int32       nJYZT;      //����״̬
    int64       BrowIndex;           //��ҳ��ѯ��λ��

} ITPDK_RQZQ;

//�ͻ������ʽ���Ϣ
typedef struct
{
    char        AccountId[16];       //�ͻ���
    int32       nTCXZ;      //ͷ�����ʣ�1��ͨͷ�� 2ר��ͷ�磩
    char        szTCBH[21];  //ͷ����
    double      dTCGM;      //ͷ���ģ
    double      dYRCJE;     //���ڳ����
    double      dYYJE;      //ԤԼ���
    double      dRCDJJE;    //�����ڳ�������
    double      dDRCHJE;    //���ճ������
    double      dDJJE;      //��ʱ������
    double      dKYED;      //���ý��

} ITPDK_KRZJXX;

//�����ʲ�
typedef struct
{
    char   AccountId[16];                //�ͻ���
    double dKYBZJ;       //���ñ�֤��
    double dDBBL;       //��������
    double dYJBL1;       //Ԥ��ά�ֵ�������1�����ǵ����ﻮ�뻮��
    double dYJBL2;         //Ԥ��ά�ֵ�������2�����ǵ����ﻮ���������ǻ���
    double dZHYE;       //�˻����
    double dKYZJ;       //�����ʽ�
    double dQSZJ;       //�����ʽ�
    double dKMDBPZJ;	//���򵣱�Ʒ�ʽ�

    double dZQSZ_BD;    //���֤ȯ��ֵ
    double dZQSZ;		    //֤ȯ��ֵ_���㵣������ʹ��
    double dZQSZ_DB;    //����֤ȯ��ֵ
    double dZQSZ_DBZS;	//����֤ȯ������ֵ_������ñ�֤��ʹ��(�ѿ�����)
    double dZQSZ_DB_BD;//����֤ȯ�䶯������ֵ_������ñ�֤��ʹ��
    double dZQSZ_DBZS_BD;//����֤ȯ�䶯������ֵ_������ñ�֤��ʹ��
    double dZQSZ_YJZC;          //�����ﷵ��δ�ɽ�_Ԥ���ʲ�

    double dRZFZ;       //���ʸ�ծ
    double dRQFZ;       //��ȯ��ծ
    double dRZFZ2;      //���ʸ�ծ2
    double dRQFZ2;      //��ȯ��ծ2
    double dHQWT2;      //��ȯί��δ��
    double dXJCE;       //Ӧ��׷���������ߵ��ʽ���
    double dPCJE;       //ǿ��ƽ���ʽ� = (�ܸ�ծ*ƽ�ֵ�λ���� - ���ʲ�)/(ƽ�ֵ�λ���� -1)
    double dZDPCJE;     //����ƽ���ʽ� = (�ܸ�ծ*׷����λ���� - ���ʲ�)/(׷����λ���� -1)
    double dKQBZ;       //�����ʲ���׼
    double dKQZJ;       //��ȡ�ʽ�

    double dRZLX;       //��Ƿ��Ϣ_����
    double dRQLX;       //��Ƿ��Ϣ_��ȯ
    double dYJLX;       //Ԥ����Ϣ/����
    double dZFZ;        //�ܸ�ծ
    double dZZC;        //���ʲ�
    double dJZC;        //���ʲ�
    //double dZYBZJ;    //ռ�ñ�֤��
    double dZYBZJ_RZ;   //ռ�ñ�֤��_����
    double dZYBZJ_RQ;   //ռ�ñ�֤��_��ȯ
    double dRZYK;       //����ӯ��
    double dRQYK;       //��ȯӯ��
    double dRQSYZJ;     //��ȯʣ���ʽ�
    double dRZSXF;      //���ʷ���
    double dRQSXF;      //��ȯ����
    double dDBSZ_SH;    //�Ϻ�����֤ȯ��ֵ
    double dDBSZ_SZ;    //���ڵ���֤ȯ��ֵ
    double dRZFYZS;     //���ʸ�ӯ����
    double dRZFK;       //���ʸ���
    double dRQFYZS;     //��ȯ��ӯ����
    double dRQFK;       //��ȯ����
    double dRQMCJE;     //��ȯ���_��ծ

    double dRZDQSZ;     //���ʵ�ǰ��ֵ
    double dRQDQSZ;     //��ȯ��ǰ��ֵ

    double dSYZQSZ;     //����֤ȯ��ֵ
						
						
						// FALG ��1  ��չ�ֶ�
    int    nJSRQ;		//��ͬ������
    double dRZED;		//�������Ŷ��
    double dRQED;       //��ȯ���Ŷ��
    double dRZZY;       //�������ö��
    double dRQZY;       //��ȯ���ö��
    double dHTED;       //��ͬ���
    double dQTZY;		//����ռ��
    double dKYDBZJ;		//���򵣱�Ʒ�ʽ�
    double dQTFY;       //��������
    double dZZCJZD;     //�ֲ�ռ���ʲ����ж�
    double dJZCJZD;     //���ʲ����ж�
    double dZZCJZD_BK;  //�ƴ���+��ҵ����ֲ�ռ���ʲ����ж�
    double dJZCJZD_BK;  //�ƴ���+��ҵ���龻�ֲּ��ж�
    double dZZCJZD_BKCDR;//�ƴ���+��ҵ��+�ƴ���CDR+��ҵ��CDR���ֲ�ռ���ʲ����ж�

    double dRZFZ_PT;       //���ʸ�ծ(��ͨͷ��)
    double dRZSXF_PT;      //���ʷ���(��ͨͷ��)
    double dRZLX_PT;       //��Ƿ��Ϣ_����(��ͨͷ��)

    double dRQMCQSJE;      //��ȯ���������������ʵ�ʳɽ���ͳ�ƣ�
    double dBY1;           //����1
    double dBY2;           //����2
   
} ITPDK_XYZC;
//���ú�Լ
typedef struct
{
    char    AccountId[16];             //�ͻ���
    char    m_szHYBH[25];  // HYBH ��Լ���
    char    m_szHTBH[17];  // HTBH ��ͬ���
    int   	m_nWTH;        // WTH	�������
    int   	m_nFSRQ;       // FSRQ	��������
    char    m_szFSSJ[13];  // FSSJ ����ʱ��
    int     m_nJYRQ;       // JYRQ  ��������
    int     m_nDQRQ;		// ��������
    int     m_nHYQX;        // ��Լ����
    int     m_nQXRQ;        // ��Ϣ����
    int     m_nHYLB;        // ��Լ���
    int     m_nHYLX;        // ��Լ����
    char    m_szJYS[3];    // JYS ������
    char    m_szGDH[11];      //�ɶ���
    char    m_szZQDM[11];  // ZQDM ֤ȯ����
    char    m_szZQMC[21];  // ZQMC ֤ȯ����
    int     m_nWTSL;       // WTSL ί������
    double  m_dWTJG;     // WTJG	���м۸�
    int     m_nCJSL;       // CJSL	�ɽ�����
    double  m_dCJJE;     // CJJE	�ɽ����
    double  m_dCJJG;     // CJJG 	�ɽ��۸�
    double  m_dSXF;      // SXF	�ɽ�������
    double  m_dYCHJE;    // YCHJE	 �ѳ�����Լ���
    double  m_dYCHSXF;   // YCHSXF �ѳ������׷���
    int     m_nYCHSL;      // YCHSL �ѳ�������
    double  m_dYCHLX;    // YCHLX	�ѳ�����Ϣ
    double  m_dYCHFX;    // YCHFX	�ѳ�����Ϣ
    double  m_dQCHYJE;   // QCHYJE	�ڳ���Լ���
    double  m_dQCSXF;    // QCSXF	�ڳ���Լ���׷���
    double  m_dQCLX;     // QCLX	�ڳ���Լ��Ϣ
    double  m_dQCFX;     // QCFX	�ڳ���Ϣ���
    int     m_nQCSL;       // QCSL	�ڳ���Լ����
    double  m_dSSHYJE;   // SSHYJE	ʵʱ��Լ���
    double  m_dSSSXF;    // SSSXF	ʵʱ��Լ���׷���
    double  m_dSSLX;     // SSLX	ʵʱ��Լ��Ϣ
    double  m_dSSFX;     // SSFX	ʵʱ��Ϣ���
    int     m_nSSSL;       // SSSL	ʵʱ��Լ����
    double  m_dHYLL;     // HYLL	����/����
    double  m_dFXFL;     // FXFL	��Ϣ����
    double  m_dLXJS;     // LXJS	��Ϣ����
    double  m_dFDLX;     // FDLX	�ֶ���Ϣ
    double  m_dYJLX;     // YJLX	Ԥ����Ϣ
    double  m_dFXLXJS;   // FXLXJS	��Ϣ��Ϣ����
    double  m_dFXYJLX;   // FXYJLX	��ϢԤ����Ϣ
    double  m_dFXFDLX;   // FXFDLX	��Ϣ�ֶ���Ϣ
    double  m_dBZJBL;    // BZJBL	��֤�����
    int     m_nZQCS;       // ZQCS	չ�ڴ���
    int     m_nEDDJBS;     // EDDJBS��ȶ����ʶ
    char    m_szYHYBH[25];           // YHYBH	ԭ��Լ���
    char    m_szTCBH[21];             // TCBH	ͷ����
    int     m_nBCSL;       // BCSL	��������
    double  m_dBCJE;     // BCJE	�������
    int     m_nLXLJFS;     // LXLJFS	��Ϣ�˽᷽ʽ
    int     m_nFLBH;       // FLBH	���ʱ��
    char    m_szDQRQHYBH[29];        // DQRQHYBH
    int     m_nQKBZ;       // QKBZ Ƿ���־
    int     m_nQQBZ;       // QQBZ	Ƿȯ��־
    int     m_nZT;			//��鸺ծ�˽�״̬��0-δ�˽�  1-���˽�
    char    m_szBrowIndex[25];//��ҳ����ֵ
} ITPDK_XYFZ;
//�������ú�Լ�䶯
typedef struct
{
    char    AccountId[16];       //�ͻ���
    int     m_nLSH;        // LSH ��ˮ��
    char    m_szHYBH[25];  // HYBH ��Լ���
    int     m_nWTH;        // WTH	�������
    int     m_nHBXH;       // HBXH  �ر����
    char    m_szJYS[3];    // JYS	������
    char    m_szZQDM[11];  // ZQDM	֤ȯ����
    int     m_nJYRQ;       // JYRQ	��������
    int     m_nFSRQ;       // FSRQ	��������
    char    m_szFSSJ[13];  // FSSJ	����ʱ��
    int     m_nHYBDLB;     // HYBDLB ��Լ�䶯���
    int     m_nJYLB;       // JYLB	�������
    int     m_nHYLB;       // HYLB	��Լ���
    char    m_szHYZQDM[11];// HYZQDM	��Լ֤ȯ����
    char    m_szHYJYS[3];  // HYJYS		��Լ������
    int     m_nFSHYSL;     // FSHYSL	������Լ����
    int     m_nHYSL;       // HYSL		��������
    double  m_dFSHYJE;     // FSHYJE	������Լ���
    double  m_dHYJE;       // HYJE		������
    double  m_dFSFY;       // FSFY		��������
    double  m_dHYFY;       // HYFY		�������
    double  m_dFSLX;       // FSLX		������Ϣ
    double  m_dHYLX;       // HYLX		������Ϣ
    double  m_dFSFX;       // FSFX		������Ϣ
    double  m_dHYFX;       // HYFX		���෣Ϣ
    int     m_nZDBZ;       // ZDBZ		ָ�������־
    int     m_nQPBZ;       // QPBZ		ǿƽ��־
    char    m_szYHYBH[25];    // YHYBH	ԭ��Լ���
    char    m_szTCBH[21];      // TCBH	ͷ����
    char    m_szZY[129];      // ZY		ժҪ
    int     m_nCXBZ;         // CXBZ	������־
    int     m_nQSBZ;         // QSBZ	�����־
} ITPDK_XYDRBD;
//����������ȯ��ծ����
typedef struct
{
    char    AccountId[16];       //�ͻ���
    char    m_szJYS[3];         //������
    char    m_szZQDM[11];       //֤ȯ����
    int     m_nFZSL;            //��ծ����
    int     m_nDRRQCJSL;        //������ȯ�ɽ�����
    int     m_nHQWTSL;          //��ȯί������
    int     m_nHQCJSL;          //��ȯ�ɽ�����
    int     m_nYQWTSL;          //��ȯί������
    int     m_nYQSL;            //��ȯ����
    int     m_nBrowIndex;       //��ҳ����
} ITPDK_XYRQFZHZ;

//A�ɿɻ���������ѯ
typedef struct
{
	char    AccountId[16];  //�ͻ���
	char    m_szJYS[3];     //������
	char    m_szGDH[12];     //�ɶ���
	char    m_szZQDM[11];   //֤ȯ����
	char    m_szZQMC[21];	//֤ȯ����
	int64     m_nKWTSL;     //�ɻ�������
	int64     m_nKMCSL;     //����������
	int64     m_nMRCJSL;    //������������

} ITPDK_XYDBHRSL;

//ר��ͷ��֤ȯ����
typedef struct
{
    int64   m_nSQH;         //�����
    int64   m_nLSH;         //��ˮ��
    int64   m_nJCCL;        //��ֲ���
    int64   m_nKMCSL;       //����������
    int64   m_nBDSL;        //�䶯����
    int64   m_nFSSL;        //��������
}ITPDK_ZXTCZQDB;

//��ѯ֤ȯ������
typedef struct
{
    int64   m_nTYPE;        //����
    char    m_szJYS[3];     //������
    char    m_szZQZBZ[64];  //֤ȯָ��ֵ ,��������֤ȯ����
    int64   m_nJYLB;        //�������
    char    m_szZQLB[4];    //֤ȯ���
    char    m_szXYSX[4];    //�������� "01" - ���ж�  "02" - ������
    char    m_szXYSXZ[64];  //��������ֵ���������ǽ������ƣ����������ƴ�ӵ��ַ���
}ITPDK_ZQHMD;

//�����¹��깺
typedef struct
{
    char     m_szJYS[4];        //������
    char     m_szZQDM[12];      //֤ȯ����
    char     m_szZQMC[16];      //֤ȯ����
    int      m_nFXFS;           //���з�ʽ
    double   m_dRGJG;           //�Ϲ��۸�
    char     m_szRGDM[12];      //�Ϲ�����
    int      m_nRGRQ;           //�Ϲ�����
    int      m_nWTSX;           //ί������
    int      m_nBDRQ;           //�䶯����
} ITPDK_XGSG;

//��ծ�䶯��ϸ
typedef struct
{
    char    AccountId[16];      // KHH      �ͻ���
    int     m_nLSH;             // LSH      ��ˮ��
    char    m_szHYBH[25];       // HYBH     ��Լ���
    int     m_nWTH;             // WTH	    �������
    int     m_nHBXH;            // HBXH     �ر����
    char    m_szJYS[3];         // JYS	    ������
    char    m_szZQDM[11];       // ZQDM	    ֤ȯ����
    int     m_nJYRQ;            // JYRQ	    ��������
    int     m_nFSRQ;            // FSRQ	    ��������
    char    m_szFSSJ[13];       // FSSJ	    ����ʱ��
    int     m_nHYBDLB;          // HYBDLB   ��Լ�䶯���
    int     m_nJYLB;            // JYLB	    �������
    int     m_nHYLB;            // HYLB	    ��Լ���
    char    m_szHYZQDM[11];     // HYZQDM	��Լ֤ȯ����
    char    m_szHYJYS[3];       // HYJYS	��Լ������
    int     m_nFSHYSL;          // FSHYSL	������Լ����
    int     m_nHYSL;            // HYSL		��������
    double  m_dFSHYJE;          // FSHYJE	������Լ���
    double  m_dHYJE;            // HYJE		������
    double  m_dFSFY;            // FSFY		��������
    double  m_dHYFY;            // HYFY		�������
    double  m_dFSLX;            // FSLX		������Ϣ
    double  m_dHYLX;            // HYLX		������Ϣ
    double  m_dFSFX;            // FSFX		������Ϣ
    double  m_dHYFX;            // HYFX		���෣Ϣ
    int     m_nZDBZ;            // ZDBZ		ָ�������־
    int     m_nQPBZ;            // QPBZ		ǿƽ��־
    char    m_szYHYBH[25];      // YHYBH	ԭ��Լ���
    char    m_szTCBH[21];       // TCBH	    ͷ����
    char    m_szZY[129];        // ZY		ժҪ
    int     m_nCXBZ;            // CXBZ	    ������־
    int     m_nQSBZ;            // QSBZ	    �����־
    int     m_nBrowIndex;       // BrwoIndex ��ҳ����
} ITPDK_FZBDMX;

//���ʲֵ���ϸ��ѯ�����
typedef struct
{
    char    m_sKCLSH[13];       //������ˮ��
    char    m_sKCRQ[9];         //��������
    char    m_sCDDQR[9];        //�ֵ�������
    char    m_sZQDM[9];         //֤ȯ����
    char    m_sZQMC[9];         //֤ȯ����
    double  m_dRZMRJE;          //����������
    double  m_dYCHJE;           //�ѳ������
    double  m_dRZFZJE;          //���ʸ�ծ���
    double  m_dRZMRJ;           //���������
    double  m_dRZFZSZ;          //���ʸ�ծ��ֵ
    double  m_dRZLXYJ;          //������Ϣ�Ѽ�
    double  m_dRZLXYH;          //������Ϣ�ѻ�
    double  m_dYQFXYJ;          //���ڷ�Ϣ�Ѽ�
    double  m_dYQFXYH;          //���ڷ�Ϣ�ѻ�
    int     m_nCDZQCS;          //�ֵ�չ�ڴ���
    double  m_dHYNLL;           //��Լ������
    char    m_sLJBS[2];         //�˽��ʶ
    double  m_dWJXF;            //δ��Ϣ��
    double  m_dYJWFXF;          //�ѽ�δ��Ϣ��
    double  m_dRZSL;            //��������
}ITPDK_RZCDMX;

//��ȯ�ֵ���ϸ��ѯ�����
typedef struct
{
    char    m_sKCLSH[13];       //������ˮ��
    char    m_sKCRQ[9];         //��������
    char    m_sCDDQR[9];        //�ֵ�������
    char    m_sZQDM[9];         //֤ȯ����
    char    m_sZQMC[9];         //֤ȯ����
    int     m_dRQMCGS;          //��ȯ��������
    int     m_dYCHGS;           //�ѳ�������
    double  m_dRQFZGS;          //��ȯ��ծ����
    double  m_dRQMCJ;           //��ȯ������
    double  m_dRQFZSZ;          //��ȯ��ծ��ֵ
    double  m_dRQFYYJ;          //��ȯ�����Ѽ�
    double  m_dRQFYYH;          //��ȯ�����ѻ�
    double  m_dYQFXYJ;          //���ڷ�Ϣ�Ѽ�
    double  m_dYQFXYH;          //���ڷ�Ϣ�ѻ�
    char    m_sQYBCFS[30];      //Ȩ�油����ʽ(��һλ��ʾ����ɳ�����־�����ڶ�λ��ʾ����ծ������־��������λ��ʾ���ɷ�Ȩ֤������־)
                                //��־���壺0��ո�:��ǰ�˽�  1:�ֽ𲹳�
    int     m_nCDZQCS;          //�ֵ�չ�ڴ���
    double  m_dHYNLL;           //��Լ������
    double  m_dWJXF;            //δ��Ϣ��
    double  m_dYJWFXF;          //�ѽ�δ��Ϣ��
}ITPDK_RQCDMX;

//�ֵ��������ܾ�ԭ���ѯ�����
typedef struct
{
    char    m_sSQRQ[9];         //��������
    char    m_sKHH[16];         //�ͻ���
    char    m_sSQXH[13];        //�������
    char    m_sCDLSH[13];       //�ֵ���ˮ��
    char    m_sCDFSR[9];        //�ֵ�������
    char    m_sCDLX[2];         //�ֵ����ͣ�1:����  2:��ȯ��
    char    m_sTZLX[2];         //�������ͣ�0:���ʵ���  1:�ֵ�չ�ڣ�
    char    m_sYYB[4];          //Ӫҵ��
    char    m_sJYS[3];          //�г�����
    char    m_sZQDM[9];         //֤ȯ����
    char    m_sJJYY[255];       //�ܾ�ԭ��
    char    m_sLSSCSJ[27];      //��ˮ����ʱ��
}ITPDK_CDBGSQJJYYCX;

//�ֵ���������ѯ�����
typedef struct
{
    char    m_sSQRQ[9];         //��������
    char    m_sKHH[16];         //�ͻ���
    char    m_sSQLSH[11];       //������ˮ��
    char    m_sTZLX[2];         //�������ͣ�0:���ʵ���  1:�ֵ�չ�ڣ�
    char    m_sHYLX[2];         //��Լ���ͣ�1:����  2:��ȯ��
    char    m_sYYB[4];          //Ӫҵ��
    char    m_sSQZG[7];         //����ְ��
    char    m_sSQSJ[9];         //����ʱ��
    char    m_sSHZG[7];         //���ְ��
    char    m_sSHZT[2];         //���״̬��0:������  1:ͨ��  2:�ܾ�  3:����ʧ�ܣ�
    char    m_sSHRQ[9];         //�������
    char    m_sSHSJ[9];         //���ʱ��
    char    m_sKCKSRQ[9];       //���ֿ�ʼ����
    char    m_sKCJSRQ[9];       //���ֽ�������
    char    m_sJYS[3];          //������
    char    m_sZQDM[9];         //֤ȯ����
    int     m_dCDBS;            //�ֵ�����
    int     m_dCLBS;            //�������
    char    m_sFJSJ[41];        //�������ݣ����ʵ���ʱ����²�Ʒ��ţ�չ��ʱ���Ԥչ��������
    char    m_sWTLY[81];        //ί����Դ
    char    m_sBZ[256];         //��ע�����״̬Ϊ 3:����ʧ�� ʱ��¼ʧ��ԭ��
}ITPDK_CDBGSQCX;

//�¹���ǩ��ѯ�����
typedef struct
{
    char    m_sFSRQ[9];         //��������
    char    m_sKHYYB[5];        //����Ӫҵ��
    char    m_sKHH[11];         //�ͻ���
    char    m_sGDH[16];         //�ɶ�����
    char    m_sGDXM[21];        //�ɶ�����
    char    m_sSCDM[2];         //�г�����
    char    m_sSCMC[21];        //�г�����
    char    m_sZQDM[9];         //֤ȯ����
    char    m_sZQMC[41];        //֤ȯ����
    int     m_nZQSL;            //��ǩ����
    double  m_dCJJG;            //�ɽ��۸�
    double  m_dCJJE;            //�ɽ����
    int     m_nZQGS;            //��ǩ����
    char    m_sLSH[31];         //��ˮ��
    char    m_sKKRQ[9];         //�ۿ�����
    char    m_sZQLB[4];         //֤ȯ���00:������01::��Ʊ��02:ծȯ��
    char    m_sCLZT[3];         //����״̬��0:δ����1:ȫ����ǩ�ۿ2:������ǩ�ۿ3:ȫ��������
    char    m_sTQDJBZ[2];       //��ǰ�����־��0:δԤ���ᣬ1:Ԥ���ᣬ2:��ǩ����
    double  m_dSJZQSL;          //ʵ����ǩ����
    double  m_dSJZQJE;          //ʵ����ǩ���
    double  m_dFQSL;            //��������
    double  m_dFQJE;            //�������
}ITPDK_XGZQ;

//��Ų�ѯ�����
typedef struct
{
    char    m_sSCDM[2];         //�г�����
    char    m_sSCMC[21];        //�г�����
    char    m_sGDH[16];         //�ɶ��˺�
    char    m_sPHDM[9];         //��Ŵ���
    char    m_sPHMC[41];        //�������
    char    m_sQSPH[13];        //��ʼ���
    int     m_nPHGS;            //��Ÿ���
    char    m_sPHRQ[9];         //�������
    char    m_sLSH[31];         //��ˮ��
    char    m_sCLXX[256];       //������Ϣ
}ITPDK_PHCX;

//��ѯ��֤ҵ����ˮ�����
typedef struct
{
    char    m_sYWLB[3];         //ת�����
    char    m_sYWLBMC[21];      //ת���������
    double  m_dFSJE;            //���
    char    m_sCLJG[13];        //ת�˷��ش���
    char    m_sJGSM[256];       //ת��ȷ����Ϣ
    char    m_sLSH[16];         //��ˮ��
    char    m_sYHMC[81];        //��������
    char    m_sYHDM[7];         //���д���
    char    m_sWTSJ[9];         //ת��ʱ��
    char    m_sWTRQ[9];         //��������
}ITPDK_YZYWLS;

//��ѯ���д��루���
typedef struct
{
    char    m_sYHDM[7];         //���д���
    char    m_sYHJC[81];        //���м��
    char    m_sYHZH[41];        //�����˺�
    char    m_sYHBZ[22];        //���б�־
    char    m_sHBDM[4];         //���Ҵ���
    char    m_sZZYMMXX[2];      //֤ȯת��������ѡ�0:�����䣬1:ֻ���ʽ����룬2:ֻ���������룬3:�������붼Ҫ�䣩
    char    m_sYZZMMXX[2];      //����ת֤ȯ����ѡ�0:�����䣬1:ֻ���ʽ����룬2:ֻ���������룬3:�������붼Ҫ�䣩
    char    m_sZHH[11];         //��/���˻���
    char    m_sZHLB[2];         //�˻����0:���˻���1:���˻���
    char    m_sZHLBMC[31];      //�˻��������
    int     m_nKSQX;            //�Ƿ�֧�ֿ���ȡ�֣�0:��֧�֣�1:֧�֣�
}ITPDK_YHDM;

//��ѯ���������
typedef struct
{
    char    m_sLSH[16];         //��ˮ��
    double  m_dYHYE;            //�������           
}ITPDK_YHYE;

//��ѯ�ͻ���Ϣ
typedef struct
{
    char    m_sKHH[16];         //�ͻ���
    char    m_sKHXM[32];        //�ͻ�����
    char    m_sYYB[8];          //Ӫҵ��
    int     m_nWTFS;            //ί�з�ʽ
}ITPDK_KHXX;

//��ѯ�ʽ������ϸ
typedef struct
{
    char    m_sKHH[16];         //�ͻ���
    int64   m_nLSH;             //��ˮ��
    char    m_sZJZH[15];        //�ʽ��˺�
    char    m_sBZ[4];           //����
    int     m_nFSRQ;            //��������
    char    m_sFSSJ[13];        //����ʱ��
    int     m_nDJLB;            //�������
    double  m_dDJJE;            //������
    int64   m_nBrowIndex;       //��ҳ����ֵ
    char    m_sZY[61];          //ժҪ
}ITPDK_ZJDJMX;

//A5�ʸ�֤ȯ
typedef struct
{
    char    m_sJYS[4];          //������
    char    m_sZQDM[12];        //֤ȯ����
    char    m_sZQMC[25];        //֤ȯ����
    int     m_nRZZT;            //����״̬
    int     m_nBZRZZT;          //��׼����״̬
    double  m_dBZRZBL;          //��׼���ʱ���
    double  m_dRZBL;            //���ʱ�֤�����
    int     m_nRQZT;            //��ȯ״̬
    int     m_nBZRQZT;          //��׼��ȯ״̬
    double  m_dBZRQBL;          //��׼��ȯ����
    double  m_dRQBL;            //��ȯ��֤�����
    int     m_nZQQYBZ;          //����չ��
    int     m_nZQCS;            //���չ�ڴ���
    int     m_nQTSX;            //��������
    int     m_nFXKZSX;          //���տ�������
    int     m_nYXRQ;            //��Ч����
    int     m_nDJRQ;            //�Ǽ�����
    int     m_nBDRQ;            //�޸����� 
    char    m_sBrowIndex[10];   //��ҳ��ѯ��λ��
} ITPDK_A5ZGZQ;

//A5����֤ȯ
typedef struct
{
    char    m_sJYS[4];          //������
    char    m_sZQDM[12];        //֤ȯ����
    char    m_sZQMC[25];        //֤ȯ����
    int     m_nJYZT;            //����״̬
    int     m_nBZJYZT;          //��׼����״̬
    double  m_dDBZSL;           //����������
    double  m_dBZZSL;           //��׼������
    int     m_nGYQYBZ;          //�Ƿ����ù��ʼ�
    double  m_dGYJG;            //���ʼ۸�
    int     m_nQTSX;            //��������
    int     m_nFXKZSX;          //���տ�������
    int     m_nYXRQ;            //��Ч����
    int     m_nDJRQ;            //�Ǽ�����
    int     m_nBDRQ;            //�޸����� 
    char    m_sBrowIndex[10];   //��ҳ��ѯ��λ��
} ITPDK_A5DBZQ;

/////////////////////////////�ӿڳ���νṹ/////////////////////////////
//�ӿ����� - �ͻ���Ϣ
struct ITPDK_CusReqInfo
{
    char     AccountId[16];       //�ͻ���
    char     FundAccount[16];      //�ʽ��
    char     SecuAccount[12];     //�ɶ���
    char     Password[40];        //��������
    char     OrgCode[8];          //��������
    int      TradeNodeID;         //���׽ڵ�
    char     EntrustWay[5];       //ί�з�ʽ
    char     NodeInfo[256];       //����վ��
    char     OppSecuAccount[12];      //�Է��ɶ���
    char     OppSeat[12];           //�Է�ϯλ
    int64    Token;               //��¼����

    int64    RetCode;             //����ֵ
    char     ErrMsg[256];         //������Ϣ

    ITPDK_CusReqInfo()
        :AccountId{ 0 }
        , FundAccount{ 0 }
        , SecuAccount{ 0 }
        , Password{ 0 }
        , OrgCode{ 0}
        , TradeNodeID(-1)
        , EntrustWay{ 0 }
        , NodeInfo{ 0 }
        , OppSecuAccount{ 0 }
        , OppSeat{ 0 }
        , Token(-1)
        , RetCode(0)
        , ErrMsg{ 0 }
    {
    }
};

};  //MarItpdk
#pragma pack(pop)
