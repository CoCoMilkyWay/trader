﻿/*!
 * \file WtMinImpactExeUnit.h
 *
 * \author Wesley
 * \date 2020/03/30
 *
 * 最小冲击执行单元
 */
#pragma once
#include "../Includes/ExecuteDefs.h"
#include "WtOrdMon.h"

USING_NS_WTP;

class WtMinImpactExeUnit : public ExecuteUnit
{
public:
	WtMinImpactExeUnit();
	virtual ~WtMinImpactExeUnit();

private:
	void	do_calc();

public:
	/*
	 *	所属执行器工厂名称
	 */
	virtual const char* getFactName() override;

	/*
	 *	执行单元名称
	 */
	virtual const char* getName() override;

	/*
	 *	初始化执行单元
	 *	ctx		执行单元运行环境
	 *	code	管理的合约代码
	 */
	virtual void init(ExecuteContext* ctx, const char* stdCode, WTSVariant* cfg) override;

	/*
	 *	订单回报
	 *	localid	本地单号
	 *	code	合约代码
	 *	isBuy	买or卖
	 *	leftover	剩余数量
	 *	price	委托价格
	 *	isCanceled	是否已撤销
	 */
	virtual void on_order(uint32_t localid, const char* stdCode, bool isBuy, double leftover, double price, bool isCanceled) override;

	/*
	 *	tick数据回调
	 *	newTick	最新的tick数据
	 */
	virtual void on_tick(WTSTickData* newTick) override;

	/*
	 *	成交回报
	 *	code	合约代码
	 *	isBuy	买or卖
	 *	vol		成交数量,这里没有正负,通过isBuy确定买入还是卖出
	 *	price	成交价格
	 */
	virtual void on_trade(uint32_t localid, const char* stdCode, bool isBuy, double vol, double price) override;

	/*
	 *	下单结果回报
	 */
	virtual void on_entrust(uint32_t localid, const char* stdCode, bool bSuccess, const char* message) override;

	/*
	 *	设置新的目标仓位
	 *	code	合约代码
	 *	newVol	新的目标仓位
	 */
	virtual void set_position(const char* stdCode, double newVol) override;

	/*
	 *	清理全部持仓
	 *	stdCode	合约代码
	 */
	virtual void clear_all_position(const char* stdCode) override;

	/*
	 *	交易通道就绪回调
	 */
	virtual void on_channel_ready() override;

	/*
	 *	交易通道丢失回调
	 */
	virtual void on_channel_lost() override;

private:
	WTSTickData* _last_tick;	//上一笔行情
	double		_target_pos;	//目标仓位
	StdUniqueMutex	_mtx_calc;

	WTSCommodityInfo*	_comm_info;
	WTSSessionInfo*		_sess_info;

	//////////////////////////////////////////////////////////////////////////
	//执行参数
	int32_t		_price_offset;
	uint32_t	_expire_secs;//订单超时秒数
	int32_t		_price_mode;
	uint32_t	_entrust_span; //发单时间间隔
	bool		_by_rate;
	double		_order_lots;//单次发单手数
	double		_qty_rate;

	/*
	 *	By Wesley @ 2022.12.15
	 *	增加一个最小开仓数量
	 *	为什么没有最小平仓数量呢，因为平仓要根据持仓来，所以无法限制
	 */
	double		_min_open_lots;//最小开仓数量

	WtOrdMon	_orders_mon;
	uint32_t	_cancel_cnt;//在途撤单量
	uint32_t	_cancel_times;//撤单次数


	uint64_t	_last_place_time;//上个下单时间
	uint64_t	_last_tick_time;//上个tick时间

	std::atomic<bool>	_in_calc;

	typedef struct _CalcFlag
	{
		bool				_result;
		std::atomic<bool>*	_flag;
		_CalcFlag(std::atomic<bool>* flag) :_flag(flag)
		{
			_result = _flag->exchange(true, std::memory_order_acq_rel);
		}

		~_CalcFlag()
		{
			if (_flag)
				_flag->exchange(false, std::memory_order_acq_rel);
		}

		operator bool() const { return _result; }
	} CalcFlag;
};

