﻿/*!
* \file MfMocker.h
* \project	WonderTrader
*
* \author Wesley
* \date 2020/03/30
*
* \brief
*/
#pragma once
#include <sstream>
#include "HisDataReplayer.h"

#include "../Includes/FasterDefs.h"
#include "../Includes/ISelStraCtx.h"
#include "../Includes/SelStrategyDefs.h"
#include "../Includes/WTSDataDef.hpp"
#include "../Share/fmtlib.h"
#include "../Share/DLLHelper.hpp"

class SelStrategy;

USING_NS_WTP;

class HisDataReplayer;

class SelMocker : public ISelStraCtx, public IDataSink
{
public:
	SelMocker(HisDataReplayer* replayer, const char* name, int32_t slippage = 0, bool isRatioSlp = false);
	virtual ~SelMocker();

private:
	template<typename... Args>
	void log_debug(const char* format, const Args& ...args)
	{
		const char* buffer = fmtutil::format(format, args...);
		stra_log_debug(buffer);
	}

	template<typename... Args>
	void log_info(const char* format, const Args& ...args)
	{
		const char* buffer = fmtutil::format(format, args...);
		stra_log_info(buffer);
	}

	template<typename... Args>
	void log_error(const char* format, const Args& ...args)
	{
		const char* buffer = fmtutil::format(format, args...);
		stra_log_error(buffer);
	}

private:
	void	dump_outputs();
	void	dump_stradata();
	inline void log_signal(const char* stdCode, double target, double price, uint64_t gentime, const char* usertag = "");
	inline void	log_trade(const char* stdCode, bool isLong, bool isOpen, uint64_t curTime, double price, double qty, const char* userTag = "", double fee = 0.0);
	inline void	log_close(const char* stdCode, bool isLong, uint64_t openTime, double openpx, uint64_t closeTime, double closepx, double qty,
		double profit, double maxprofit, double maxloss, double totalprofit = 0, const char* enterTag = "", const char* exitTag = "", uint32_t openBarNo = 0, uint32_t closeBarNo = 0);

	void	update_dyn_profit(const char* stdCode, double price);

	void	do_set_position(const char* stdCode, double qty, double price = 0.0, const char* userTag = "", bool bTriggered = false);
	void	append_signal(const char* stdCode, double qty, const char* userTag = "", double price = 0.0);

	void	proc_tick(const char* stdCode, double last_px, double cur_px);

public:
	bool	init_sel_factory(WTSVariant* cfg);

public:
	//////////////////////////////////////////////////////////////////////////
	//IDataSink
	virtual void	handle_tick(const char* stdCode, WTSTickData* curTick, uint32_t pxType) override;
	virtual void	handle_bar_close(const char* stdCode, const char* period, uint32_t times, WTSBarStruct* newBar) override;
	virtual void	handle_schedule(uint32_t uDate, uint32_t uTime) override;

	virtual void	handle_init() override;
	virtual void	handle_session_begin(uint32_t uCurDate) override;
	virtual void	handle_session_end(uint32_t uCurDate) override;
	virtual void	handle_replay_done() override;

	//////////////////////////////////////////////////////////////////////////
	//ICtaStraCtx
	virtual uint32_t id() { return _context_id; }

	//回调函数
	virtual void on_init() override;
	virtual void on_session_begin(uint32_t curTDate) override;
	virtual void on_session_end(uint32_t curTDate) override;
	virtual void on_tick(const char* stdCode, WTSTickData* newTick, bool bEmitStrategy = true) override;
	virtual void on_bar(const char* stdCode, const char* period, uint32_t times, WTSBarStruct* newBar) override;
	virtual bool on_schedule(uint32_t curDate, uint32_t curTime, uint32_t fireTime) override;
	virtual void enum_position(FuncEnumSelPositionCallBack cb) override;

	virtual void on_tick_updated(const char* stdCode, WTSTickData* newTick) override;
	virtual void on_bar_close(const char* stdCode, const char* period, WTSBarStruct* newBar) override;
	virtual void on_strategy_schedule(uint32_t curDate, uint32_t curTime) override;


	//////////////////////////////////////////////////////////////////////////
	//策略接口
	virtual double stra_get_position(const char* stdCode, bool bOnlyValid = false, const char* userTag = "") override;
	virtual void stra_set_position(const char* stdCode, double qty, const char* userTag = "") override;
	virtual double stra_get_price(const char* stdCode) override;

	/*
	 *	读取当日价格
	 */
	virtual double stra_get_day_price(const char* stdCode, int flag = 0) override;

	virtual uint32_t stra_get_tdate() override;
	virtual uint32_t stra_get_date() override;
	virtual uint32_t stra_get_time() override;

	virtual double stra_get_fund_data(int flag = 0) override;

	virtual uint64_t stra_get_first_entertime(const char* stdCode) override;
	virtual uint64_t stra_get_last_entertime(const char* stdCode) override;
	virtual uint64_t stra_get_last_exittime(const char* stdCode) override;
	virtual double stra_get_last_enterprice(const char* stdCode) override;
	virtual const char* stra_get_last_entertag(const char* stdCode) override;
	virtual double stra_get_position_avgpx(const char* stdCode) override;
	virtual double stra_get_position_profit(const char* stdCode) override;

	virtual uint64_t stra_get_detail_entertime(const char* stdCode, const char* userTag) override;
	virtual double stra_get_detail_cost(const char* stdCode, const char* userTag) override;
	virtual double stra_get_detail_profit(const char* stdCode, const char* userTag, int flag = 0) override;

	virtual WTSCommodityInfo* stra_get_comminfo(const char* stdCode) override;
	virtual WTSSessionInfo* stra_get_sessinfo(const char* stdCode) override;
	virtual WTSKlineSlice*	stra_get_bars(const char* stdCode, const char* period, uint32_t count) override;
	virtual WTSTickSlice*	stra_get_ticks(const char* stdCode, uint32_t count) override;
	virtual WTSTickData*	stra_get_last_tick(const char* stdCode) override;

	/*
	 *	获取分月合约代码
	 */
	virtual std::string		stra_get_rawcode(const char* stdCode) override;

	virtual void stra_sub_ticks(const char* stdCode) override;

	virtual void stra_log_info(const char* message) override;
	virtual void stra_log_debug(const char* message) override;
	virtual void stra_log_warn(const char* message) override;
	virtual void stra_log_error(const char* message) override;

	virtual void stra_save_user_data(const char* key, const char* val) override;

	virtual const char* stra_load_user_data(const char* key, const char* defVal = "") override;

protected:
	uint32_t			_context_id;
	HisDataReplayer*	_replayer;

	uint64_t		_total_calc_time;	//总计算时间
	uint32_t		_emit_times;		//总计算次数
	int32_t			_slippage;			//成交滑点
	bool			_ratio_slippage;	//是否比例滑点
	uint32_t		_schedule_times;	//调度次数

	std::string		_main_key;

	typedef struct _KlineTag
	{
		bool		_closed;
		uint32_t	_count;

		_KlineTag() :_closed(false), _count(0){}

	} KlineTag;
	typedef wt_hashmap<std::string, KlineTag> KlineTags;
	KlineTags	_kline_tags;

	typedef std::pair<double, uint64_t>	PriceInfo;
	typedef wt_hashmap<std::string, PriceInfo> PriceMap;
	PriceMap		_price_map;

	typedef struct _DetailInfo
	{
		bool		_long;
		double		_price;
		double		_volume;
		uint64_t	_opentime;
		uint32_t	_opentdate;
		double		_max_profit;
		double		_max_loss;
		double		_max_price;
		double		_min_price;
		double		_profit;
		char		_opentag[32];
		uint32_t	_open_barno;

		_DetailInfo()
		{
			memset(this, 0, sizeof(_DetailInfo));
		}
	} DetailInfo;

	typedef struct _PosInfo
	{
		double		_volume;
		double		_closeprofit;
		double		_dynprofit;
		uint64_t	_last_entertime;
		uint64_t	_last_exittime;
		double		_frozen;

		std::vector<DetailInfo> _details;

		_PosInfo()
		{
			_volume = 0;
			_closeprofit = 0;
			_dynprofit = 0;
			_frozen = 0;
			_last_entertime = 0;
			_last_exittime = 0;
		}

		inline double valid() const { return _volume - _frozen; }
	} PosInfo;
	typedef wt_hashmap<std::string, PosInfo> PositionMap;
	PositionMap		_pos_map;

	typedef struct _SigInfo
	{
		double		_volume;
		std::string	_usertag;
		double		_sigprice;
		double		_desprice;
		bool		_triggered;
		uint64_t	_gentime;

		_SigInfo()
		{
			_volume = 0;
			_sigprice = 0;
			_desprice = 0;
			_triggered = false;
			_gentime = 0;
		}
	}SigInfo;
	typedef wt_hashmap<std::string, SigInfo>	SignalMap;
	SignalMap		_sig_map;

	std::stringstream	_trade_logs;
	std::stringstream	_close_logs;
	std::stringstream	_fund_logs;
	std::stringstream	_sig_logs;
	std::stringstream	_pos_logs;

	//是否处于调度中的标记
	bool			_is_in_schedule;	//是否在自动调度中

	//用户数据
	typedef wt_hashmap<std::string, std::string> StringHashMap;
	StringHashMap	_user_datas;
	bool			_ud_modified;

	typedef struct _StraFundInfo
	{
		double	_total_profit;
		double	_total_dynprofit;
		double	_total_fees;

		_StraFundInfo()
		{
			memset(this, 0, sizeof(_StraFundInfo));
		}
	} StraFundInfo;

	StraFundInfo		_fund_info;

	typedef struct _StraFactInfo
	{
		std::string		_module_path;
		DllHandle		_module_inst;
		ISelStrategyFact*		_fact;
		FuncCreateSelStraFact	_creator;
		FuncDeleteSelStraFact	_remover;

		_StraFactInfo()
		{
			_module_inst = NULL;
			_fact = NULL;
		}

		~_StraFactInfo()
		{
			if (_fact)
				_remover(_fact);
		}
	} StraFactInfo;
	StraFactInfo	_factory;

	SelStrategy*	_strategy;

	uint32_t		_cur_tdate;

	//tick订阅列表
	wt_hashset<std::string> _tick_subs;
};