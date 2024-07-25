﻿#include "WtShareHelper.h"
#include "ShareBlocks.h"

using namespace shareblock;

bool init_master(const char* id, const char* path/* = ""*/)
{
	return ShareBlocks::one().init_master(id, path);
}


bool init_slave(const char* id, const char* path/* = ""*/)
{
	return ShareBlocks::one().init_slave(id, path);
}

bool update_slave(const char* id, bool bForce/* = false*/)
{
	return ShareBlocks::one().update_slave(id, bForce);
}

bool release_slave(const char* name)
{
	return ShareBlocks::one().release_slave(name);
}

uint32_t get_sections(const char* domain, FuncGetSections cb)
{
	auto ay = ShareBlocks::one().get_sections(domain);
	for (const std::string& v : ay)
		cb(v.c_str());

	return (uint32_t)ay.size();
}

uint32_t get_keys(const char* domain, const char* section, FuncGetKeys cb)
{
	auto ay = ShareBlocks::one().get_keys(domain, section);
	for (KeyInfo* v : ay)
		cb(v->_key, v->_type);

	return (uint32_t)ay.size();
}

uint64_t get_section_updatetime(const char* domain, const char* section)
{
	return ShareBlocks::one().get_section_updatetime(domain, section);
}

bool commit_section(const char* domain, const char* section)
{
	return ShareBlocks::one().commit_section(domain, section);
}

bool delete_section(const char* domain, const char*section)
{
	return ShareBlocks::one().delete_section(domain, section);
}

const char* allocate_string(const char* domain, const char* section, const char* key, const char* initVal, bool bForceWrite /*= false*/)
{
	return ShareBlocks::one().allocate_string(domain, section, key, initVal, bForceWrite);
}

int32_t* allocate_int32(const char* domain, const char* section, const char* key, int32_t initVal, bool bForceWrite /*= false*/)
{
	return ShareBlocks::one().allocate_int32(domain, section, key, initVal, bForceWrite);
}

int64_t* allocate_int64(const char* domain, const char* section, const char* key, int64_t initVal, bool bForceWrite /*= false*/)
{
	return ShareBlocks::one().allocate_int64(domain, section, key, initVal, bForceWrite);
}

uint32_t* allocate_uint32(const char* domain, const char* section, const char* key, uint32_t initVal, bool bForceWrite /*= false*/)
{
	return ShareBlocks::one().allocate_uint32(domain, section, key,  initVal, bForceWrite);
}

uint64_t* allocate_uint64(const char* domain, const char* section, const char* key, uint64_t initVal, bool bForceWrite /*= false*/)
{
	return ShareBlocks::one().allocate_uint64(domain, section, key, initVal, bForceWrite);
}

double* allocate_double(const char* domain, const char* section, const char* key, double initVal, bool bForceWrite /*= false*/)
{
	return ShareBlocks::one().allocate_double(domain, section, key, initVal, bForceWrite);
}

bool set_string(const char* domain, const char* section, const char* key, const char* val)
{
	return ShareBlocks::one().set_string(domain, section, key, val);
}

bool set_int32(const char* domain, const char* section, const char* key, int32_t val)
{
	return ShareBlocks::one().set_int32(domain, section, key, val);
}

bool set_int64(const char* domain, const char* section, const char* key, int64_t val)
{
	return ShareBlocks::one().set_int64(domain, section, key, val);
}

bool set_uint32(const char* domain, const char* section, const char* key, uint32_t val)
{
	return ShareBlocks::one().set_uint32(domain, section, key, val);
}

bool set_uint64(const char* domain, const char* section, const char* key, uint64_t val)
{
	return ShareBlocks::one().set_uint64(domain, section, key, val);
}

bool set_double(const char* domain, const char* section, const char* key, double val)
{
	return ShareBlocks::one().set_double(domain, section, key, val);
}

const char* get_string(const char* domain, const char* section, const char* key, const char* defVal /* = "" */)
{
	return ShareBlocks::one().get_string(domain, section, key, defVal);
}

int32_t get_int32(const char* domain, const char* section, const char* key, int32_t defVal /* = 0 */)
{
	return ShareBlocks::one().get_int32(domain, section, key, defVal);
}

int64_t get_int64(const char* domain, const char* section, const char* key, int64_t defVal /* = 0 */)
{
	return ShareBlocks::one().get_int64(domain, section, key, defVal);
}

uint32_t get_uint32(const char* domain, const char* section, const char* key, uint32_t defVal /* = 0 */)
{
	return ShareBlocks::one().get_uint32(domain, section, key, defVal);
}

uint64_t get_uint64(const char* domain, const char* section, const char* key, uint64_t defVal /* = 0 */)
{
	return ShareBlocks::one().get_uint64(domain, section, key, defVal);
}

double get_double(const char* domain, const char* section, const char* key, double defVal /* = 0 */)
{
	return ShareBlocks::one().get_double(domain, section, key, defVal);
}

bool init_cmder(const char* name, bool isCmder /* = false */, const char* path /* = "" */)
{
	return ShareBlocks::one().init_cmder(name, isCmder, path);
}

bool add_cmd(const char* name, const char* cmd)
{
	return ShareBlocks::one().add_cmd(name, cmd);
}

const char* get_cmd(const char* name, uint32_t& lastIdx)
{
	return ShareBlocks::one().get_cmd(name, lastIdx);
}