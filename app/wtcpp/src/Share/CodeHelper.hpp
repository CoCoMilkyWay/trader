﻿/*!
 * \file CodeHelper.hpp
 * \project	WonderTrader
 *
 * \author Wesley
 * \date 2020/03/30
 * 
 * \brief 代码辅助类,封装到一起方便使用
 */
#pragma once
#include "fmtlib.h"
#include "StrUtil.hpp"
#include "../Includes/WTSTypes.h"
#include "../Includes/IHotMgr.h"

#include <boost/xpressive/xpressive_dynamic.hpp>


USING_NS_WTP;

//主力合约后缀
static const char* SUFFIX_HOT = ".HOT";

//次主力合约后缀
static const char* SUFFIX_2ND = ".2ND";

//前复权合约代码后缀
static const char SUFFIX_QFQ = '-';

//后复权合约代码后缀
static const char SUFFIX_HFQ = '+';

class CodeHelper
{
public:
	typedef struct _CodeInfo
	{
		char _code[MAX_INSTRUMENT_LENGTH];		//合约代码
		char _exchg[MAX_INSTRUMENT_LENGTH];		//交易所代码
		char _product[MAX_INSTRUMENT_LENGTH];	//品种代码
		char _ruletag[MAX_INSTRUMENT_LENGTH];	//
		char _fullpid[MAX_INSTRUMENT_LENGTH];	//

		//By Wesley @ 2021.12.25
		//去掉合约类型，这里不再进行判断
		//整个CodeHelper会重构
		//ContractCategory	_category;		//合约类型
		//union
		//{
		//	uint8_t	_hotflag;	//主力标记，0-非主力，1-主力，2-次主力
		//	uint8_t	_exright;	//是否是复权代码,如SH600000Q: 0-不复权, 1-前复权, 2-后复权
		//};

		/*
		 *	By Wesley @ 2022.03.07
		 *	取消原来的union
		 *	要把主力标记和复权标记分开处理
		 *	因为后面要对主力合约做复权处理了
		 */
		uint8_t	_exright;	//是否是复权代码,如SH600000Q: 0-不复权, 1-前复权, 2-后复权

		//是否是复权代码
		inline bool isExright() const { return _exright != 0; }

		//是否前复权代码
		inline bool isForwardAdj() const { return _exright == 1; }

		//是否后复权代码
		inline bool isBackwardAdj() const { return _exright == 2; }

		//标准品种ID
		inline const char* stdCommID()
		{
			if (strlen(_fullpid) == 0)
				fmtutil::format_to(_fullpid, "{}.{}", _exchg, _product);

			return _fullpid;
		}

		_CodeInfo()
		{
			memset(this, 0, sizeof(_CodeInfo));
			//_category = CC_Future;
		}

		inline void clear()
		{
			memset(this, 0, sizeof(_CodeInfo));
		}

		inline bool hasRule() const
		{
			return strlen(_ruletag) > 0;
		}
	} CodeInfo;

private:
	static inline std::size_t find(const char* src, char symbol = '.', bool bReverse = false)
	{
		std::size_t len = strlen(src);
		if (len != 0)
		{
			if (bReverse)
			{
				for (std::size_t idx = len - 1; idx >= 0; idx--)
				{
					if (src[idx] == symbol)
						return idx;
				}
			}
			else
			{
				for (std::size_t idx = 0; idx < len; idx++)
				{
					if (src[idx] == symbol)
						return idx;
				}
			}
		}


		return std::string::npos;
	}

public:
	/*
	 *	是否是期货期权合约代码
	 *	CFFEX.IO2007.C.4000
	 */
	static bool	isStdChnFutOptCode(const char* code)
	{
		/* 定义正则表达式 */
		//static cregex reg_stk = cregex::compile("^[A-Z]+.[A-z]+\\d{4}.(C|P).\\d+$");	//CFFEX.IO2007.C.4000
		//return 	regex_match(code, reg_stk);
		char state = 0;
		std::size_t i = 0;
		for(; ; i++)
		{
			char ch = code[i];
			if(ch == '\0')
				break;

			if(state == 0)
			{
				if (!('A' <= ch && ch <= 'Z'))
					return false;

				state += 1;
			}
			else if (state == 1)
			{
				if ('A' <= ch && ch <= 'Z')
					continue;

				if (ch == '.')
					state += 1;
				else
					return false;
			}
			else if (state == 2)
			{
				if (!('A' <= ch && ch <= 'z'))
					return false;

				state += 1;
			}
			else if (state == 3)
			{
				if ('A' <= ch && ch <= 'z')
					continue;

				if ('0' <= ch && ch <= '9')
					state += 1;
				else
					return false;
			}
			else if (state >= 4 && state <= 6)
			{
				if ('0' <= ch && ch <= '9')
					state += 1;
				else
					return false;
			}
			else if (state == 7)
			{
				if (ch == '.')
					state += 1;
				else
					return false;
			}
			else if (state == 8)
			{
				if (ch == 'C' || ch == 'P')
					state += 1;
				else
					return false;
			}
			else if (state == 9)
			{
				if (ch == '.')
					state += 1;
				else
					return false;
			}
			else if (state == 10)
			{
				if ('0' <= ch && ch <= '9')
					state += 1;
				else
					return false;
			}
			else if (state == 11)
			{
				if ('0' <= ch && ch <= '9')
					continue;
				else
					return false;
			}
		}

		return (state == 11);
	}

	/*
	 *	是否是标准分月期货合约代码
	 *	//CFFEX.IF.2007
	 */
	static inline bool	isStdMonthlyFutCode(const char* code)
	{
		using namespace boost::xpressive;
		/* 定义正则表达式 */
		static cregex reg_stk = cregex::compile("^[A-Z]+.[A-z]+.\\d{4}$");	//CFFEX.IO.2007
		return 	regex_match(code, reg_stk);
	}

	/*
	 *	标准代码转标准品种ID
	 *	如SHFE.ag.1912->SHFE.ag
	 *	如果是简化的股票代码，如SSE.600000，则转成SSE.STK
	 */
	static inline std::string stdCodeToStdCommID2(const char* stdCode)
	{
		auto idx = find(stdCode, '.', true);
		auto idx2 = find(stdCode, '.', false);
		if (idx != idx2)
		{
			//前后两个.不是同一个，说明是三段的代码
			//提取前两段作为品种代码
			return std::string(stdCode, idx);
		}
		else
		{
			//两段的代码，直接返回
			//主要针对某些交易所，每个合约的交易规则都不同的情况
			//这种情况，就把合约直接当成品种来用
			return stdCode;
		}		
	}

	/*
	 *	从基础分月合约代码提取基础品种代码
	 *	如ag1912 -> ag
	 *	这个只有分月期货品种才有意义
	 *	这个不会有永续合约的代码传到这里来，如果有的话就是调用的地方有Bug!
	 */
	static inline std::string rawMonthCodeToRawCommID(const char* code)
	{
		int nLen = 0;
		while ('A' <= code[nLen] && code[nLen] <= 'z')
			nLen++;

		return std::string(code, nLen);
	}

	/*
	 *	基础分月合约代码转标准码
	 *	如ag1912转成全码
	 *	这个不会有永续合约的代码传到这里来，如果有的话就是调用的地方有Bug!
	 */
	static inline std::string rawMonthCodeToStdCode(const char* code, const char* exchg, bool isComm = false)
	{
		thread_local static char buffer[64] = { 0 };
		std::size_t len = 0;
		if(isComm)
		{
			len = strlen(exchg);
			memcpy(buffer, exchg, len);
			buffer[len] = '.';
			len += 1;

			auto clen = strlen(code);
			memcpy(buffer+len, code, clen);
			len += clen;
			buffer[len] = '\0';
			len += 1;
		}
		else
		{
			std::string pid = rawMonthCodeToRawCommID(code);
			len = strlen(exchg);
			memcpy(buffer, exchg, len);
			buffer[len] = '.';
			len += 1;

			memcpy(buffer + len, pid.c_str(), pid.size());
			len += pid.size();
			buffer[len] = '.';
			len += 1;

			char* s = (char*)code;
			s += pid.size();
			if (strlen(s) == 4)
			{
				wt_strcpy(buffer + len, s, 4);
				len += 4;
			}
			else
			{
				if (s[0] > '5')
					buffer[len] = '1';
				else
					buffer[len] = '2';
				len += 1;
				wt_strcpy(buffer + len, s, 3);
				len += 3;
			}
		}

		return std::string(buffer, len);
	}

	/*
	 *	原始常规代码转标准代码
	 *	这种主要针对非分月合约而言
	 */
	static inline std::string rawFlatCodeToStdCode(const char* code, const char* exchg, const char* pid)
	{
		thread_local static char buffer[64] = { 0 };
		auto len = strlen(exchg);
		memcpy(buffer, exchg, len);
		buffer[len] = '.';
		len += 1;

		auto plen = strlen(pid);
		auto clen = strlen(code);

		if (strcmp(code, pid) == 0 || plen == 0)
		{
			memcpy(buffer + len, code, clen);
			len += clen;
			buffer[len] = '\0';
		}
		else
		{
			memcpy(buffer + len, pid, plen);
			len += plen;
			buffer[len] = '.';
			len += 1;

			memcpy(buffer + len, code, clen);
			len += clen;
			buffer[len] = '\0';
		}

		return buffer;
	}

	static inline bool isMonthlyCode(const char* code)
	{
		//using namespace boost::xpressive;
		//最后3-6位都是数字，才是分月合约
		//static cregex reg_stk = cregex::compile("^.*[A-z|-]\\d{3,6}$");	//CFFEX.IO.2007
		//return 	regex_match(code, reg_stk);
		auto len = strlen(code);
		char state = 0;
		for (std::size_t i = 0; i < len; i++)
		{
			char ch = code[len - i - 1];
			if (0 <= state && state < 3)
			{
				if (!('0' <= ch && ch <= '9'))
					return false;

				state += 1;
			}
			else if (3 == state || 4 == state)
			{
				if ('0' <= ch && ch <= '9')
					state += 1;

				else if ('A' <= ch && ch <= 'z')
				{
					state = 7;
					break;
				}
			}
			else if (4 == state)
			{
				if ('0' <= ch && ch <= '9')
					state += 1;

				else if (('A' <= ch && ch <= 'z') || ch == '-')
				{
					state = 7;
					break;
				}
			}
			else if (state < 6)
			{
				if ('0' <= ch && ch <= '9')
					state += 1;
				else
					return false;
			}
			else if (state >= 6)
			{
				if (('A' <= ch && ch <= 'z') || ch == '-')
				{
					state = 7;
					break;
				}
				else
				{
					return false;
				}
			}
		}

		return state == 7;
	}

	/*
	 *	期货期权代码标准化
	 *	标准期货期权代码格式为CFFEX.IO2008.C.4300
	 *	-- 暂时没有地方调用 --
	 */
	static inline std::string rawFutOptCodeToStdCode(const char* code, const char* exchg)
	{
		using namespace boost::xpressive;
		/* 定义正则表达式 */
		static cregex reg_stk = cregex::compile("^[A-z]+\\d{4}-(C|P)-\\d+$");	//中金所、大商所格式IO2013-C-4000
		bool bMatch = regex_match(code, reg_stk);
		if(bMatch)
		{
			std::string s = std::move(fmt::format("{}.{}", exchg, code));
			StrUtil::replace(s, "-", ".");
			return s;
		}
		else
		{
			//郑商所上期所期权代码格式ZC010P11600

			//先从后往前定位到P或C的位置
			std::size_t idx = strlen(code) - 1;
			for(; idx >= 0; idx--)
			{
				if(!isdigit(code[idx]))
					break;
			}
			
			std::string s = exchg;
			s.append(".");
			s.append(code, idx-3);
			if(strcmp(exchg, "CZCE") == 0)
				s.append("2");
			s.append(&code[idx - 3], 3);
			s.append(".");
			s.append(&code[idx], 1);
			s.append(".");
			s.append(&code[idx + 1]);
			return s;
		}
	}

	/*
	 *	标准合约代码转主力代码
	 */
	static inline std::string stdCodeToStdHotCode(const char* stdCode)
	{
		std::size_t idx = find(stdCode, '.', true);
		if (idx == std::string::npos)
			return "";		
		
		std::string stdWrappedCode;
		stdWrappedCode.resize(idx + strlen(SUFFIX_HOT) + 1);
		memcpy((char*)stdWrappedCode.data(), stdCode, idx);
		wt_strcpy((char*)stdWrappedCode.data()+idx, SUFFIX_HOT);
		return stdWrappedCode;
	}

	/*
	 *	标准合约代码转次主力代码
	 */
	static inline std::string stdCodeToStd2ndCode(const char* stdCode)
	{
		std::size_t idx = find(stdCode, '.', true);
		if (idx == std::string::npos)
			return "";

		std::string stdWrappedCode;
		stdWrappedCode.resize(idx + strlen(SUFFIX_2ND) + 1);
		memcpy((char*)stdWrappedCode.data(), stdCode, idx);
		wt_strcpy((char*)stdWrappedCode.data() + idx, SUFFIX_2ND);
		return stdWrappedCode;
	}

	/*
	 *	标准期货期权代码转原代码
	 *	-- 暂时没有地方调用 --
	 */
	static inline std::string stdFutOptCodeToRawCode(const char* stdCode)
	{
		std::string ret = stdCode;
		auto pos = ret.find(".");
		ret = ret.substr(pos + 1);
		if (strncmp(stdCode, "CFFEX", 5) == 0 || strncmp(stdCode, "DCE", 3) == 0)
			StrUtil::replace(ret, ".", "-");
		else
			StrUtil::replace(ret, ".", "");
		return ret;
	}

	static inline int indexCodeMonth(const char* code)
	{
		if (strlen(code) == 0)
			return -1;

		std::size_t idx = 0;
		std::size_t len = strlen(code);
		while(idx < len)
		{
			if (isdigit(code[idx]))
				return (int)idx;

			idx++;
		}
		return -1;
	}

	/*
	 *	提取标准期货期权代码的信息
	 */
	static CodeInfo extractStdChnFutOptCode(const char* stdCode)
	{
		CodeInfo codeInfo;

		StringVector ay = StrUtil::split(stdCode, ".");
		wt_strcpy(codeInfo._exchg, ay[0].c_str());
		if(strcmp(codeInfo._exchg, "SHFE") == 0 || strcmp(codeInfo._exchg, "INE") == 0)
		{
			fmt::format_to(codeInfo._code, "{}{}{}", ay[1], ay[2], ay[3]);
		}
		else if (strcmp(codeInfo._exchg, "CZCE") == 0)
		{
			std::string& s = ay[1];
			fmt::format_to(codeInfo._code, "{}{}{}{}", s.substr(0, s.size()-4), s.substr(s.size()-3), ay[2], ay[3]);
		}
		else
		{
			fmt::format_to(codeInfo._code, "{}-{}-{}", ay[1], ay[2], ay[3]);
		}

		int mpos = indexCodeMonth(ay[1].c_str());

		if(strcmp(codeInfo._exchg, "CZCE") == 0)
		{
			memcpy(codeInfo._product, ay[1].c_str(), mpos);
			strcat(codeInfo._product, ay[2].c_str());
		}
		else if (strcmp(codeInfo._exchg, "CFFEX") == 0)
		{
			memcpy(codeInfo._product, ay[1].c_str(), mpos);
		}
		else
		{
			memcpy(codeInfo._product, ay[1].c_str(), mpos);
			strcat(codeInfo._product, "_o");
		}

		return codeInfo;
	}

	/*
	 *	提起标准代码的信息
	 */
	static CodeInfo extractStdCode(const char* stdCode, IHotMgr *hotMgr)
	{
		//期权的代码规则和其他都不一样，所以单独判断
		if(isStdChnFutOptCode(stdCode))
		{
			return extractStdChnFutOptCode(stdCode);
		}
		else
		{
			/*
			 *	By Wesley @ 2021.12.25
			 *	1、先看是不是Q和H结尾的，如果是复权标记确认以后，最后一段长度-1，复制到code，如SSE.STK.600000Q
			 *	2、再看是不是分月合约，如果是，则将product字段拼接月份给code（郑商所特殊处理），如CFFEX.IF.2112
			 *	3、最后看看是不是HOT和2ND结尾的，如果是，则将product拷贝给code，如DCE.m.HOT
			 *	4、如果都不是，则原样复制第三段，如BINANCE.DC.BTCUSDT/SSE.STK.600000
			 */
			thread_local static CodeInfo codeInfo;
			codeInfo.clear();
			auto idx = StrUtil::findFirst(stdCode, '.');
			wt_strcpy(codeInfo._exchg, stdCode, idx);

			auto idx2 = StrUtil::findFirst(stdCode + idx + 1, '.');
			if (idx2 == std::string::npos)
			{
				wt_strcpy(codeInfo._product, stdCode + idx + 1);

				//By Wesley @ 2021.12.29
				//如果是两段的合约代码，如OKEX.BTC-USDT
				//则品种代码和合约代码一致
				wt_strcpy(codeInfo._code, stdCode + idx + 1);
			}
			else
			{
				wt_strcpy(codeInfo._product, stdCode + idx + 1, idx2);
				const char* ext = stdCode + idx + idx2 + 2;
				std::size_t extlen = strlen(ext);
				char lastCh = ext[extlen - 1];
				if (lastCh == SUFFIX_QFQ || lastCh == SUFFIX_HFQ)
				{
					codeInfo._exright = (lastCh == SUFFIX_QFQ) ? 1 : 2;

					extlen--;
					lastCh = ext[extlen - 1];
				}
				
				if (extlen == 4 && '0' <= lastCh && lastCh <= '9')
				{
					//如果最后一段是4位数字，说明是分月合约
					//TODO: 这样的判断存在一个假设，最后一位是数字的一定是期货分月合约，以后可能会有问题，先注释一下
					//那么code得加上品种id
					//郑商所得单独处理一下，这个只能hardcode了
					auto i = wt_strcpy(codeInfo._code, codeInfo._product);
					if (memcmp(codeInfo._exchg, "CZCE", 4) == 0)
						wt_strcpy(codeInfo._code + i, ext + 1, extlen-1);
					else
						wt_strcpy(codeInfo._code + i, ext, extlen);
				}
				else
				{
					const char* ruleTag = (hotMgr != NULL) ? hotMgr->getRuleTag(ext) :"";
					if (strlen(ruleTag) == 0)
						wt_strcpy(codeInfo._code, ext, extlen);
					else
					{
						wt_strcpy(codeInfo._code, codeInfo._product);
						wt_strcpy(codeInfo._ruletag, ruleTag);
					}
				}
			}			

			return codeInfo;
		}
	}
};

