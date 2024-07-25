﻿/*!
 * \file EventCaster.h
 * \project	WonderTrader
 *
 * \author Wesley
 * \date 2020/03/30
 * 
 * \brief UDP广播对象定义
 */
#pragma once

#include <queue>

#include "../Includes/WTSMarcos.h"
#include "../Share/StdUtils.hpp"

NS_WTP_BEGIN
class MQManager;

class MQServer
{
public:
	MQServer(MQManager* mgr);
	~MQServer();

public:
	inline uint32_t id() const { return _id; }

	bool	init(const char* url, bool confirm = false);

	void	publish(const char* topic, const void* data, uint32_t dataLen);

private:
	std::string		_url;
	bool			_ready;
	int				_sock;
	MQManager*		_mgr;
	uint32_t		_id;
	bool			_confirm;

	StdThreadPtr	m_thrdCast;
	StdCondVariable	m_condCast;
	StdUniqueMutex	m_mtxCast;
	bool			m_bTerminated;
	bool			m_bTimeout;

	typedef struct _PubData
	{
		std::string	_topic;
		std::string	_data;

		_PubData(const char* topic, const void* data, uint32_t dataLen)
			: _topic(topic)
		{
			if(data !=  NULL && dataLen != 0)
			{
				_data.append((const char*)data, dataLen);
			}
		}
	} PubData;
	typedef std::queue<PubData> PubDataQue;

	PubDataQue		m_dataQue;
	std::string		m_sendBuf;
};

NS_WTP_END